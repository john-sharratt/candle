# Switching to the hybrid: recurrent state, glue, and provenance

Status: **partly implemented**. Scope is the hybrid Qwen3.5/3.6/3.8 lineage
(`candle-transformers/src/models/delta_net/`, `models/qwen35/`) becoming the
production model in place of Qwen3-30B-A3B.

## Implementation status

Phases 0, 1, 2, 3, 4, 5a, 5b, 5c, 5d, 5e, 7 and 9 are **implemented and green**,
and **the hybrid now runs end to end through zend on real weights.** §11a records
what the design got wrong along the way — including three defects the first live
run exposed (§11a.6–§11a.8).

**Tier 3 passes — all of P9.** The daemon loads the 22 GB checkpoint, ingests,
seals a complete 30-layer non-zero recurrent snapshot per turn, carries state
across nine turns (it recalled a colour stated on turn 1), and survives a
process restart resuming from the redo log. 281 s for the three.

**Tier 2 passes on the model hooks.** The `<think>` oracle (T3.4) — a discarded
thinking pass leaves the sealed state **byte-identical**, which is site 8 proven
fixed rather than argued fixed. `decode_replay_probe` reports 0/7 dirty on the
hybrid (T3.8), so the harness control is honest again. `fork_recurrent` is an
independent copy (T1.6).

**Still open:** P6 needs a new per-leaf prefill pass (§11a.3b) and is the one
remaining item that is a feature rather than a measurement. P8 is conditional on
a seal-latency measurement. P10 is measurements. T5b.5 is not implementable as
written (§11a.10).

**Turns now own their boundaries at both ends.** A turn's grid is
`[user_start][/no_think?][user_msg][user_end][assistant_start][response][im_end]`,
the assembler emits nothing around a sealed turn, and the live turn's opener is
ordinary tail prefill rather than a reserved island. `assemble_pieces` no longer
takes the markers or the `no_think` look-up at all — the boundary decision left
the assembler entirely, which is visible in its signature.

That removes the glue that **grows with the conversation** (one island per turn
boundary). What remains is the fixed system-prompt glue — template sections,
`TreeGlue` markers, `member_glue` — which is P5c.

**The feasibility gate is now enforced rather than discovered.**
`ModelCoreProperties::can_gap_fill` is `false` for the hybrid, and
`reserve_glue_island` refuses up front with a message naming what asked for an
island, instead of the wave bailing on `n_glue > 0` a layer down after the
projection was planned around it.

**The correctness gate is met.** `state_advances_across_turn_boundaries` and
`view_disposition_moves_on_reprojection_and_discards_on_clean_reprefill` both
pass, so the defect that silently ran the model on a quarter of its stack is
fixed and pinned in both directions.

**The feasibility gate is not.** A projection still emits glue, so the wave will
still bail on `n_glue > 0`. Nothing in phases 5a–5c has landed.

Three things were found during implementation that the design got wrong, all in
§11a: the rewind cannot be deleted the way §5.2 said, phase 5a is **not**
independently landable, and the hybrid was never wired into the model registry
at all — a prerequisite for the switch that appears nowhere in the 142-item list.

## Abstract

The engine's substrate is built on one idea: a conversation's history lives on
disk as sealed KV, and is **spliced back rather than re-prefilled**. Every
feature that matters — unbounded context, forking, restart, shared prefixes
across developer forks — is a consequence of that idea.

A hybrid model breaks it. Three quarters of Qwen3.6-35B's layers mix tokens
through a **recurrent state** (a delta-rule matrix plus a conv tail, per layer,
per sequence) instead of a KV cache. Tokens whose KV is reused never pass
through the model again, which is correct for attention and meaningless for a
recurrence: a recurrence has no per-token record to splice, and its memory of
those tokens exists only as an accumulated matrix that nothing today saves,
copies, or carries forward.

Tracing that through the engine turned up three problems, in increasing order of
how much they cost to find:

1. **Nothing carries the recurrent state across a turn.** The turn loop already
   carves a child slot per turn — a *view* — and decodes on it. State is keyed by
   sequence id, a view has its own, so it starts at zero while its KV borrows the
   parent's whole history; at the end, only the KV moves back. On a hybrid the
   recurrent layers therefore carry nothing in either direction, and the model
   runs on its ten attention layers. **This gates the switch.**
2. **This lineage cannot gap-fill.** Reprojection glue computes KV for tokens
   inserted mid-sequence, which a recurrence cannot do even in principle — token
   *t*'s output depends on the accumulated state over everything before it, in
   order. Five separate producers of live glue have to become sealed content
   instead. **This gates feasibility:** until a projection reserves zero gap
   chunks, the wave bails outright.
3. **Provenance signatures are hardcoded to a 48-layer, 4-KV-head, head_dim-128
   stack.** Four constants misbehave on a 10-layer, 2-head, 256-dim one — two of
   them silently, including one that disables a GPU fast path and falls back to a
   full device→host copy per seal.

The through-line is the failure mode, and it is why §6 is a test *strategy*
rather than a test list: **every one of these is fluent, shape-correct, and
error-free.** A model with a zeroed recurrent state reads perfectly and has
simply forgotten; a mis-folded signature returns a confident number. This is the
same signature as the three defects that cost the hybrid its original bring-up
(`qwen35_qwen38_models.md` §7.8), none of which was visible as a crash.

It reaches the instruments too, which is the part worth bracing for. The replay
probe that exists to separate harness drift from model non-determinism rolls back
KV and not state, so on this lineage it manufactures the drift it is there to
rule out (§3 defect 8); the provenance layer-attribution harness silently scores
an empty projection once the fold shape moves (§4.8). Both look like results.
Assume no existing measurement of the hybrid means what it says until its
instrument has been re-read.

The good news is that most of the machinery already exists and is untested only
because nothing calls it: the snapshot record format, its single-tail supersede,
the compaction race filter, `export`/`import` with hash and geometry validation,
and a device-to-device state copy. What is missing is callers, four derived
constants, and one architectural change — teaching views to carry state.

## How to read this

- **Starting work?** Read the abstract, then **§0** (pre-reading), then **§11**
  (the TODO). §0 is also what to re-read after a session compaction.
- **Reviewing the design?** §1 (the problem) → §3 (what is missing) → §4
  (design) → §9/§10 (open questions and their defaults).
- **Looking for a specific decision?** §10 lists every question with the default
  to build against.

Structure: §1–§3 state the problem and inventory what exists. §4 is the design,
including the two areas the work grew into — glue (§4.6–§4.7d) and provenance
(§4.8). §5 covers truncation, which turns out to be an operation this
architecture should not have at all. §6 is the three-tier test strategy. §7–§8
are cost and phasing, §9–§10 the open questions and their defaults, and §11 the
complete ordered work list.

## 0a. RESUME STATE — read this first after a compaction

Live checkpoint of an in-progress implementation run. Everything below is fact
about the tree as it stands, not plan.

### 0a.1 The environment (verified, not assumed)

- **GPU**: RTX PRO 5000 Blackwell, 73,415 MiB, ~71 GiB free. `nvidia-smi` works.
- **The 35B checkpoint is already on disk**, at the exact revision
  `models/qwen36_moe.rs` pins:
  `~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
  (22,134,528,992 bytes).

**So the tier-2 and tier-3 tests are NOT blocked.** An earlier pass in this run
recorded them as "needs the 22 GB checkpoint" without checking whether it was
already there. It is. Run them.

### 0a.2 What is done

**The definition of done is met.** All four of its conditions hold: the
correctness gate (T2.1/T2.2), the feasibility gate (T5c.1), no fold regression
on the outgoing model (T5e.4), and zend ingest/converse/restart (P9.1–P9.3).

Phases 0, 1, 2, 3, 4, 5a, 5b, 5c, 5d, 5e, 7, 9 — implemented; **997
candle-conversation lib tests green**, `clippy -D warnings` clean on the
workspace and on the cuda crates, `fmt --check` clean.

Verified on real weights, not asserted:

| Gate | What it says |
|------|--------------|
| P9.1 | a complete 30-layer **non-zero** snapshot per sealed turn |
| P9.2 | state advances across nine turns; recalled a colour stated on turn 1 |
| P9.3 | a fresh process resumes the timeline from the redo log and continues |
| T3.4 | a discarded thinking pass leaves the sealed state **byte-identical** |
| T3.8 | `decode_replay_probe` 0/7 dirty — the harness control is honest |
| T1.6 | `fork_recurrent` is an independent copy, not an alias |
| T7.6 | a fork and its parent continue identically under argmax |
| T5e.4 | Qwen3-30B still folds under the locked constants |
| T5e.5 | the hybrid fills all three groups: `[8,1,1]`, shift 64, 6×256 bits |
| T5d.2 | the GPU sign-pack path is *taken* at head_dim 256 |

The hybrid is wired into the model registry (`ModelArch::Qwen35Hybrid`,
`Model::Qwen36_35B_A3B_Q4`, the builder arm, the `qwen35moe` arch-string
mapping) and that loader now has run — see §11a.7 for what it took.

### 0a.3 What is left

Every phase is now implemented or deliberately closed. What remains is
measurement, and it is measurement of the research kind.

1. **P10.1 / P10.2 / P10.4** — experiments needing a labelled retrieval corpus
   and an analysis pass, not code (§11a.12). This campaign made them runnable
   rather than running them.
2. **P10.5** — re-opened by measurement: snapshots cost **62.8 MiB per turn**,
   the largest recurring write in the system. bf16 halves it; the quality
   question is §7's.
3. **P10.6** — re-opened by measurement: **three** state forks per turn, not
   one, so ~188 MiB/turn moves device-to-device. Find the other two carves.
4. **P6.2, P8, P10.3, T5b.5** — closed, each with a reason: no cross-product to
   accelerate (§11a.11), 4 % of wall (§11a.12), the capability refusal already
   makes the hybrid safe (§11a.3c), and not implementable as written (§11a.10).

### 0a.4 Decisions already taken — do not re-litigate

§11a records seven corrections with evidence. The ones most likely to be
re-opened by mistake:

- The rewind is **unreachable via a capability gate**, not deleted (§11a.1).
  `truncate_sequence_to_tokens` still exists and has legitimate non-rewind
  callers.
- P5a/P5b landed **together** and the bake is **central** in
  `Sequence::submit_prefill_unit` (§11a.2). Do not move it per-path.
- The fold check compares **group sizes only** (§11a.1 / P5e notes). Checking
  shape-derived fields makes test geometries contaminate each other and would
  refuse correct comparisons in production.
- P8.1 and P10.3 were deliberately not written blind (§11a.3c) — that reasoning
  is now void for P8.1, since the measurement can be taken.

### 0a.5 Standing constraints

`CLAUDE.md` governs: no back-compat shims, no env flags, no TODOs or stubs, one
concern per file, imports not fully-qualified paths, **only Edit/Write may
modify files**, and **never commit without explicit permission**. The whole run
so far is uncommitted on branch `fused-kernels`.

## 0. Pre-reading — what to load before starting

Read this **after** finishing the document for the first time, and **again after
any session compaction**. The design below is a set of claims about code; the
claims are not actionable until the code they describe is in context.

Two tiers. **Tier A is mandatory every time** — ~3,700 lines, all of it small
files read whole. **Tier B is per-phase** — large files where only named ranges
matter, loaded when you start that phase and not before. Reading all of Tier B
up front is ~23,000 lines and will not survive a compaction anyway.

A note on ranges: line numbers throughout this document come from a survey and
**drift**. Every range below is paired with the symbol it brackets. If a range
does not contain its symbol, trust the symbol and re-locate — do not read the
range and assume.

### 0.1 Tier A — always, whole files

| file | lines | why |
|---|---|---|
| `CLAUDE.md` (repo root) | — | the standing rules this work is bound by: no back-compat shims, no env flags, no TODOs/stubs, one concern per file, TDD, **only Edit/Write may modify files**, never commit without permission |
| `candle-transformers/src/models/delta_net/state_store.rs` | 615 | `RecurrentStateStore` — the thing being persisted, forked and restored. `export`/`import`, and the **ping-pong** wave discipline (two `s` buffers per layer; commit is a `mem::swap`, rollback is nothing). **P1, P2, P4, P7 all edit this.** |
| `candle-transformers/src/models/delta_net/types.rs` | 58 | `DeltaNetDims`, `LayerKind` — the geometry every size calculation uses |
| `candle-transformers/src/models/delta_net/kv_layout.rs` | 220 | `KvLayerMap` — why `session.num_layers()` is 10 and `model.num_layers()` is 40; `snap_to_attention` |
| `candle-transformers/src/models/qwen35/batched.rs` | 511 | the recurrent map, `ensure_recurrent` (the `offset == 0` reset, `:175`), `take/put_recurrent`, `release_recurrent` (`:214`), the wave begin/commit/rollback |
| `candle-transformers/src/models/qwen35/engine.rs` | 303 | `provenance_layer_indices` (already hybrid-correct), `create_session` |
| `candle-conversation/src/turn_layout.rs` | 873 | `TurnSegment` / `GlueKind` / `KvSpan` / `from_flat_grid` — **the whole of §4.7 lives here**, including the already-tested `user_content_start > 0` path |
| `candle-conversation/src/provenance/mod.rs` | 50 | the pipeline in one page: index side at seal, query side in decode |
| `candle-conversation/src/persistence/streams.rs` | 293 | `TurnDecl` — a turn IS a stream; `segments` is the persisted layout |

Also read, from `mix.rs` (do **not** read all 1,873 lines):

- **`mix.rs:1-40`** — the recurrence in three lines, and the layer around it.
- **`mix.rs:45-110`** — `DeltaNetState`, `snapshot()`, `copy_from()`, and the
  comment explaining why it is deliberately not `Clone`. P1's `fork_from` is
  built directly on `snapshot()`. Read both doc comments carefully: since the
  ping-pong landed, **neither is on the wave path any more** — `snapshot` serves
  the reference model's bookkeeping and `copy_from` is explicitly labelled "the
  wave path does not use this." A fork is now their only hot caller.
- **`mix.rs:165-345`** — `delta_chunked`, for the `v_new = u − w·Sᵀ` line that
  §4.6 uses to prove state does not compose.

**Companion design docs** (skim, do not read whole):
`docs/qwen35_qwen38_models.md` — **§5** (the original snapshot design), **§7.8**
(the three invisible defects — the standing warning behind §6's test strategy),
**§7.16** (F32 accumulation is mandatory), **§8 risk 1 and risk 5**.

### 0.2 Tier B — per phase, named ranges only

Load these when you start the phase, not before.

**Before P0 / P2 / P7 (the hooks and the scheduler wiring)**
- `candle-transformers/src/models/batched_inference.rs`
  — `:1533-1560` `create_view_sequence` (a view is a new `create_sequence()`)
  — `:2852-2900` `ProvenanceLayerIndices`, `ModelCoreProperties`
  — `:2995-3160` the `ManagedBatchedModel` trait head + `model_core_properties`
  — `:3240-3270` `truncate_sequence` (the shape every new hook copies)
  — `:3460-3500` the spec-decode accept path — `truncate_sequence`'s only caller
- `candle-conversation/src/scheduler/mod.rs`
  — `:1940-1960` the `Scheduler` struct (confirms `model` is reachable)
  — `:2470-2530` `handle_request` — `NewSequence` / `NewEphemeralSequence` /
    `ResumeSequence` all funnelling to `create_sequence`
  — `:2960-3040` view registration (`turn_views.insert`) and the
    `FreeSequence` handler
  — `:5320-5375` `finalize_view` at seal
  — `:6690-6745` `create_view`
  — `:7500-7560` the reprojection path: free the view, rebuild the parent,
    carve a fresh one
  — `:7790-8110` the test harness: `DummyModel`, `make_test_session`,
    `make_test_scheduler` — **P-INFRA extends exactly this**
- `candle-conversation/src/conversation.rs` — `:2455-2545` `fork`,
  `fork_resuming`, `fork_onto`

**Before P3 (truncation removal)** — §5.1's table is 14 rows across two
primitives; read the non-zero ones, not just the scheduler's.
- `candle-conversation/src/scheduler/mod.rs` — `:4640-4660`, `:5478-5500` (the
  `<think>` partial truncate — the scheduler's only non-zero target),
  `:5610-5640`, `:5720-5760`, `:7540-7560`
- `candle-conversation/src/scheduler/projection_assembler.rs` — `:605-640`
  (`apply_segments_build`)
- `candle-conversation/src/scheduler/prefill.rs` — `:2440-2490`
- `candle-nn/src/kv_cache/chunked/sequence_ops.rs` —
  `truncate_sequence_to_blocks` / `_to_tokens` (~:1900-2100)
- `candle-transformers/src/models/batched_inference.rs` — `:1180-1245`
  (`reserve_glue_gap`'s rollback, site 9), `:1251-1290` (both session wrappers),
  `:1385-1410` (the seal quantize clear, site 7)
- `candle-nn/src/kv_cache/chunked/alloc.rs` — `:1670-1735`
  (`heal_tail_divergence`, site 11) and `candle-nn/src/kv_cache/cache.rs`
  — `:695-725` (`truncate_chunked_to_tokens`, site 12)
- `candle-transformers/src/models/batch_test/utils.rs` — `:505-600`
  (`decode_replay_probe`, site 13) and `:1165-1200` (the repeat loop, site 14).
  **Read the doc comments, not only the code** — both state the invariant they
  no longer hold on a hybrid, which is what makes them §3 defect 8.

**Before P4 (snapshot writer + strip)**
- `candle-conversation/src/persistence/record.rs` — `:100-200` (`RecordType`,
  esp. `Snapshot = 20` and `Distilled = 16`), `:1008-1130` (`SnapshotPayload`)
- `candle-conversation/src/persistence/compaction.rs` — `:120-250`
  (`collect_live_records`: the snapshot loop and the `distilled` map below it)
- `candle-conversation/src/persistence/maintenance.rs` — `:610-660`
  (the `live_snapshots` supersede filter), `:1010-1360` (its four tests)
- `candle-conversation/src/persistence/mod.rs` — `:210-250`
  (`record_snapshot_loc`), `:650-720` (`track_snapshot_loc`,
  `append_recurrent_snapshot`)
- `candle-conversation/src/projection/resolver.rs` — `:2670-2710`
  (`enqueue_recurrent_snapshot`, `recurrent_snapshot_loc`)
- `candle-conversation/src/scheduler/mod.rs` — `:6250-6300` (the seal: sig
  gather, then `record_turn`), `:6380-6410` (the `WideQSig` + `Tokens`
  enqueues — where P4.4 inserts)

**Before P5a / P5b (turn boundaries)**
- `candle-conversation/src/scheduler/projection_assembler.rs` — `:150-220`
  (`BoundaryMarkers`, incl. the "intra-turn markers stay baked" note),
  `:255-335` (`assemble_pieces` — the wrapping being removed),
  `:1700-1918` (its tests, incl. the stream this must reproduce)
- `candle-conversation/src/scheduler/mod.rs` — `:2765-2860` (the live turn's
  opener + `no_think`), `:4180-4210` (the compression-turn builder — **P5b.7**),
  `:4390-4480` (`tool_exchange_segments`, the baked-marker precedent)
- `candle-conversation/src/substrate.rs` — `:735-790` (the persisted-turn
  contract, and the reason boundary markers were never baked)
- `candle-conversation/src/projection/project.rs` — `:220-270`
  (`ProjectionSegment`, `SealedKind`)

**Before P5c (system-prompt glue)**
- `candle-conversation/src/projection/schema.rs` — `:125-165`
  (`is_collection_member` — the collection approximation precedent),
  `:180-330` (`SectionTree`, `TreeDim`, `TreeNode`, `TreeGlue`),
  `:356-500` (`TreeCollection`, `TreeVariant`, `SectionCollection`,
  `member_glue`)
- `candle-conversation/src/projection/project.rs` — `:1890-1920` (`TreeGlue`
  emission), `:1970-2000` (`is_template`), `:2080-2100` (`push_member_glue`)
- `candle-conversation/src/scheduler/projection_assembler.rs` — `:640-800`
  (the island walk + prefix hash), `:980-1030` (`reserve_glue_island` — what a
  gap chunk actually is)

**Before P5d / P5e (provenance)**
- `candle-conversation/src/provenance/wide_sig.rs` (447, whole) — the fold
- `candle-conversation/src/provenance/scan.rs` — `:30-130`
  (`HEADS_PER_GROUP`, `score_provenance_late_fusion_weighted`)
- `candle-conversation/src/provenance/gpu.rs` — `:120-180`,
  `candle-conversation/src/provenance/packed.rs` — `:30-130`
  (the other two `HEADS_PER_GROUP` consumers)
- `candle-conversation/src/scheduler/mod.rs` — `:620-700`
  (`assemble_folded_prov_sigs`, the GPU fold), `:6580-6700`
  (`gather_wide_sigs` — **both** paths, incl. the CPU fallback fold at ~:6682)
- `candle-transformers/src/models/batched_inference.rs` — `:1915-1960`
  (`prov_sub_head_dim` and the `> 32` guard that silently disables the fast path,
  **plus** the `ProvSignPacked` literal above it that stamps the global
  `N_PALETTE` beside the derived `sub_head_dim` — P5d.1a)
- `zend/examples/provenance_layers.rs` — `:30-72` (the fourth, un-aliased
  `HEADS_PER_GROUP` and the `project_groups` stride it feeds — P5e.5a)
- `candle-nn/src/kv_cache/mod.rs` — `:110-260` (`active_kv_formats`, `R16` and
  its Q-capture space, `QuantFormat`)
- `candle-nn/src/kv_cache/chunked/backing.rs` — `:215-235` (`n_palette()`,
  the per-backing mechanism P5d.2 extends)

**Before P6 (branch checkpoints)**
- `candle-conversation/src/conversation.rs` — `:670-760`
  (`insert_section_collection` and friends — where variants are sealed)
- plus the P5c schema ranges above

**Before P9 (zend)**
- `zend/tests/infinite_conversation_smoke.rs` and `..._deep.rs` — the tiered
  `#[ignore]` convention and the conversation-growth harness to copy
- `zend/src/session.rs` — `:2140-2175` (`fork_resuming` at daemon resume)

### 0.3 Read-order for a first session

1. This document, end to end.
2. Tier A, in table order. `state_store.rs` and `turn_layout.rs` are the two
   that repay the closest reading — most of §4 is about one or the other.
3. `§0.2` for the phase you are starting, and only that phase.
4. Re-read **§3** (what is missing) and **§10** (decisions) with the code in
   context. Several §3 claims are counting arguments — "one caller in the whole
   tree", "zero callers" — and are worth re-verifying by grep rather than
   trusted, because they are the load-bearing facts under the phasing.

### 0.4 Verify-before-trusting

Four claims in this document are the ones everything else rests on. Each is a
single grep, and each has been wrong at some point during drafting:

```
grep -rn "\.truncate_sequence(" --include=*.rs .         # 1  — spec-decode accept, the only caller
grep -rn "enqueue_recurrent_snapshot" --include=*.rs .   # 1  — its own definition; nothing writes
grep -rn "release_recurrent" --include=*.rs .            # 2  — definition + one unreachable caller
grep -rn "fn create_view_sequence" -A 6 --include=*.rs . # 2  — session-level + backing-level; the
                                                         #      session one must call create_sequence()
grep -rn "truncate_sequence_to_tokens" --include=*.rs .  # the OTHER truncation family — §5.1
```

Counts verified against the tree at the time of writing. If one differs, the
phasing that rests on it needs re-deriving before you start:

- **`truncate_sequence` > 1** ⇒ §5.1's survey is stale and P3 is bigger than
  stated.
- **`enqueue_recurrent_snapshot` > 1** ⇒ someone landed a writer; P4 is partly
  done and §3 gap 1 is wrong.
- **`release_recurrent` > 2** ⇒ P0 may already be partly wired.
- **the session `create_view_sequence` not calling `create_sequence()`** ⇒ a
  view is no longer a distinct sequence id, which removes gap 0 entirely and
  invalidates the phase-2 gate.

The last grep has no expected count, deliberately — it is there because an
earlier draft of §5.1 surveyed only `truncate_sequence_to_blocks` and concluded
the rewind could be deleted by removing a block-count parameter. The token form
is the one that actually rewinds, it has four non-zero callers, and none of them
is reachable from the `_to_blocks` name. **Read what this returns before
starting P3**; if it returns nothing, P3.4a is already done and the rest of P3
shrinks accordingly.

If any count differs from the expectation, the phasing that depends on it needs
re-deriving before you start.

## 1. The problem

Three quarters of a hybrid stack's layers mix tokens through a **recurrent
state** rather than a KV cache. Per sequence, per DeltaNet layer, that state is
a delta-rule matrix `S [n_v_heads, d_v, d_k]` plus a causal-conv tail
`[conv_dim, K−1]`, both F32 ([mix.rs:55-61](../candle-transformers/src/models/delta_net/mix.rs#L55-L61)).

The substrate's whole design is that a conversation's history lives on disk and
is **spliced back as sealed KV chunks** rather than re-prefilled. The projection
assembler tracks this explicitly (`reused: true`, `reused_glue_tokens`). Tokens
whose KV is reused never pass through the model again — which is correct for
attention and catastrophic for a recurrence, because a recurrence has no
per-token record to splice. Its memory of those tokens exists only as the
accumulated `S`, and nothing today saves it.

The result is a conversation that resumes with attention KV covering its full
history on ¼ of its layers and `S = 0` on the other ¾. Nothing errors, every
shape matches, and the model reads fluently — it has simply forgotten
everything the recurrent layers were carrying. This is §7.8 defect 2 of
`qwen35_qwen38_models.md` wearing the opposite mask: that one was *state without
KV*, this is *KV without state*.

Affected paths:

| path | attention KV | recurrent state today |
|---|---|---|
| **every dialogue turn** (view carve — §5.3a) | borrowed from parent | **zeros** |
| **new conversation** (system prompt injected — §4.6) | injected, pre-sealed | **zeros** |
| `fork()` a live conversation | spliced from parent | **zeros** |
| daemon restart → `fork_resuming` | recovered from redo log | **zeros** |
| shared prefix across dev forks | shared | **zeros** |
| section / prefix cold-load | cold-loaded | **zeros** |
| reprojection / glue | rebuilt for the new projection | **zeros** (see below) |

The first row is the one that blocks the switch, and it was found late: the
turn machinery already carves a child slot per turn (a *view*), and the
recurrent state does not follow it. See §5.3a — on today's code the recurrent
layers carry nothing across a turn boundary in either direction.

The reprojection row needs care, because two different things are true of it.
The *intended* semantics is deliberately out of scope: we want the recurrence to
keep the original token stream and continue from there, not to be rebuilt
against the reprojected order. But **today it is not carried forward either** —
a reprojection frees the view and carves a fresh one, and a fresh view gets
zeros like every other new sequence id. So the row is a symptom of the same
gap-0 defect, not an exception to it. Once views carry state (§5.3a) the
intended behaviour falls out for free, because reprojection *finalizes*: the
state moves to the parent with the decoded blocks.

## 2. What already exists

More of this is built than the gap suggests. The following all exist, are
tested, and need no new design — only callers.

**Record format.** `RecordType::Snapshot = 20`, with `SnapshotPayload`
{`timeline_id`, `turn_index`, `schedule_hash`, `Vec<SnapshotLayer>`} and a
versioned binary codec that validates every blob length against its declared
dims ([record.rs:1008-1129](../candle-conversation/src/persistence/record.rs#L1008-L1129)).
Round-trip tested. The per-layer encoding already carries a **dtype tag byte**,
currently pinned to `0 = F32` — the designed-in extension point for §7.

**Single-tail supersede.** Snapshots are keyed in the record *header* by a
synthetic per-timeline stream id (`snapshot_stream_id`, a content hash over
`b"recurrent-snapshot"` + timeline). Last-writer-wins by append order, so the
newest snapshot supersedes every previous one **mechanically**. There is no
explicit tombstone for supersede: the accounting *is* the tombstone, crediting
the superseded copy as dead bytes.

> This answers one item from the brief directly. "When we compact we should
> automatically mark previous DeltaNet state events for tombstoning" is already
> the semantics, and it does not need a marker record — the header key gives it
> for free. See §4.

**The compaction race is already handled.** The seal thread's writer can append
a newer tail for a planned stream between plan and execute. Maintenance filters
the relocation plan against the live-tail map under the persistence lock, so a
planned-but-now-stale snapshot is skipped entirely rather than relocated after
the newer record ([maintenance.rs:619-641](../candle-conversation/src/persistence/maintenance.rs#L619-L641)).
Relocating it would have installed rolled-back state as the tail on the next
reload. That hazard is closed, with tests.

**Reload installs the location.** The load walk records each `Snapshot`'s
location per stream, and a timeline-scoped `Tombstone` removes the entry, so a
snapshot replayed before its timeline's tombstone never reads as live
([mod.rs:221-244](../candle-conversation/src/persistence/mod.rs#L221-L244)).
Both the pre-substrate walk and the runtime append path keep the map current.

**Compaction carries the live tail forward.** `collect_live_records` stages one
snapshot per conversation verbatim as a `Raw` item — the payload is a multi-MB
blob nothing holds in RAM ([compaction.rs:149-169](../candle-conversation/src/persistence/compaction.rs#L149-L169)).

**Export / import.** `RecurrentStateStore::export()` produces
`ExportedLayerState` rows field-for-field matching `SnapshotLayer`;
`import(hash, layers)` validates `schedule_hash`, layer count, and every
layer's geometry **before touching any tensor**, then writes into the slots'
existing buffers rather than replacing them
([state_store.rs:231-322](../candle-transformers/src/models/delta_net/state_store.rs#L231-L322)).
Export refuses mid-wave; import refuses mid-wave and on hash mismatch.

**The device-copy primitive.** `DeltaNetState::snapshot()` is
`Tensor::copy()` on both buffers, which resolves to `CudaStorage::try_clone` —
a device-to-device copy that never touches the host. This is already exactly
the fast VRAM fork the brief asks for, and it survives; what changed under it is
that it now has **no caller at all** on the hot path.

> **The ping-pong commit moved this ground.** The wave used to preserve its entry
> state by copying every layer aside, so a fork's copy was the same operation the
> engine already ran every wave. It no longer is: each slot holds two `s` buffers,
> a wave reads one and writes the other, commit is a host `mem::swap` and rollback
> is nothing. `copy_from`'s own doc comment now says "the wave path does not use
> this." The primitive is intact and correct, but §4.4's cost argument has to be
> re-derived rather than inherited — see there.

**The seal site.** The turn seal writes `WideQSig` then `Tokens` then fires the
persistence trigger ([scheduler/mod.rs:6388-6407](../candle-conversation/src/scheduler/mod.rs#L6388-L6407)).
The snapshot write belongs immediately beside the `WideQSig` write, before the
`Tokens` enqueue.

**The sequence-creation funnel.** `NewSequence`, `NewEphemeralSequence`, and
`ResumeSequence` all route through `Scheduler::create_sequence`
([scheduler/mod.rs:2476-2526](../candle-conversation/src/scheduler/mod.rs#L2476-L2526)).
One hook point serves both fork-copy and resume-restore.

**The per-sequence model hook precedent.** `ManagedBatchedModel::truncate_sequence`
is a per-sequence lifecycle method the hybrid overrides to *also* act on
recurrent state ([forward.rs:176-193](../candle-transformers/src/models/qwen35/forward.rs#L176-L193)).
Every new hook below (`export_recurrent`, `restore_recurrent`,
`release_sequence`) follows that *shape* — a trait method with a no-op default
that the hybrid overrides. The method itself does not survive: §5 shows its only
caller is speculative decode and its rewind capability is being deleted.

## 3. What is missing

Six gaps and three pre-existing defects. Gap 0 gates the model switch; gaps
1–5 are the feature work; defects 6–8 are live bugs on today's code,
independent of any of it.

0. **Views do not carry the state — so nothing carries it across a turn.**
   The turn loop carves a child slot (a *view*) at turn start and decodes on
   it. The view is a distinct sequence id, recurrent state is keyed by
   sequence id, and `ensure_recurrent` therefore hands the view **zeros** while
   its KV borrows the parent's whole history. `finalize_view` moves the
   decoded KV back and drops the view's state. Net effect on a hybrid: the
   recurrent layers carry nothing across a turn boundary in either direction.
   Full analysis in **§5.3a**; this is the phase-2 gate.

1. **No writer.** `enqueue_recurrent_snapshot` has exactly one hit in the
   repository: its own definition. Nothing has ever written a snapshot.
2. **No reader.** `recurrent_snapshot_loc` is never called outside persistence
   internals and tests. `import()` is never called outside `state_store`'s own
   tests. The payload is never fetched and never scattered.
3. **Distillation does not strip the state.** `keep_chunks = !turn_dead &&
   distill.is_none()` drops KV for both distill modes, but the snapshot loop
   above it has **no distill gate** — it stages every entry in
   `recurrent_snapshot_entries()` unconditionally. A distilled timeline
   therefore sheds megabytes of KV and keeps ~63 MiB of recurrent state that
   nothing can ever use. (Timeline *tombstone* does remove the entry; only
   distillation leaks.)
4. **Fork copies nothing.** `fork_onto` allocates a fresh slot and returns; the
   parent's live state is never consulted. Same root cause as gap 0 — a new
   sequence id gets a vacant store — but on an explicit user-facing fork
   rather than on every turn.
5. **The pre-generated system prompt has no state, and emits live glue.**
   A new conversation Arc-injects sealed prompt K/V, so the wave never runs
   those tokens and the state starts at zero under a full prompt. Separately,
   `TreeGlue` nodes emit `ProjectionSegment::Generated` runs — glue, inside the
   system prompt, on a lineage that cannot gap-fill (§5). Both are addressed
   in **§4.6**; unlike gap 0 this does not block the switch, it bounds how
   good the switched model is on turn one.

And three **pre-existing defects** found while tracing this, independent of the
new work:

6. **A freed slot never releases its recurrent state.** `release_recurrent` is
   called from exactly one place — `ManagedBatchedModel::truncate_sequence` —
   and that method has **one caller in the whole tree**, which is not the
   scheduler (see §5). The scheduler's free path calls
   `session.free_sequence(...)` directly and `HybridBatched` overrides no free
   hook, so the model's `HashMap<usize, RecurrentStateStore>` grows for the life
   of the process at ~63 MiB per conversation. Slot ids are pool indices and
   **are recycled**: a recycled id whose first wave carries `offset > 0` takes
   the `Occupied` arm of `ensure_recurrent` and **inherits the previous
   conversation's state** — defect 2 resurrected by slot recycling.

7. **The `<think>` clean-reprefill skews the state on every thinking turn.**
   See §5.3. This is a live-conversation corruption, not a fork/resume one.

8. **The gate harness's own replay control drifts on a hybrid.**
   `decode_replay_probe` rolls the KV back between replays and compares logits,
   and its doc comment states the purpose exactly: *"if that did not restore
   byte-identical state the probe would report its own drift as model
   non-determinism."* The probe's wave commits, so `S` advances a token per
   replay, and `truncate_sequence_to_tokens` cannot reach it — each replay
   therefore enters from a different state and the divergence it reports is the
   harness's, attributed to the model. The repeat loop of
   `test_parallel_batched_forwarding` has the same defect behind the same claim
   ("every repeat is a true re-prefill from identical state"). Both are §5.1
   sites 13–14, both are in `src/`, and both fire during the hybrid's own model
   gates — so this is an instrument that lies specifically about the lineage it
   is about to be pointed at. Fixed by P3.4e.

## 4. Design

### 4.1 Seal-path writer

At the turn seal, beside the `WideQSig` write and **before** the `Tokens`
enqueue:

```
if model exposes recurrent state:
    payload = model.export_recurrent(slot)        # new ManagedBatchedModel hook
    conversation.enqueue_recurrent_snapshot(timeline, payload.encode())
```

`export_recurrent(slot) -> Option<SnapshotPayload>` returns `None` for a
non-hybrid model, so the site is model-agnostic and Qwen3-30B pays nothing.
`turn_index` is stamped from the sealing turn; `schedule_hash` from the store.

**Ordering.** The snapshot must be durable no later than the turn it describes.
`SnapshotPayload.turn_index` already carries the rule for the torn case: reload
discards a snapshot *newer* than the last recovered turn and falls back to
recompute. Writing before the `Tokens` enqueue means a torn shutdown can leave a
snapshot for a turn whose records never landed — handled — but never a turn
whose snapshot never landed, which would be silent staleness.

**Export refuses mid-wave**, and the seal runs outside the wave, so the
precondition holds. Worth an assertion rather than an assumption.

**Cost.** This is the design's one open number — see §7.

### 4.2 Compaction: supersede and strip

*Supersede* needs no new code. The header key already makes the newest snapshot
the tail, the accounting already credits the old copy as dead, and the
plan/execute race is already filtered. Compaction drops every superseded copy
and carries exactly the tail forward. The brief's "when we compact we'll
basically remove all the previous DeltaNet's" is the existing behaviour, and the
race it worries about is the one `live_snapshots` was written to close.

*Strip* is the gap. Two triggers must remove a conversation's snapshot:

- **Distillation** (`DistillMode::{ProvenanceOnly, TextOnly}`) — the tool
  provenance corpus keeps `WideQSig` + `StreamDecl` and sheds content. The
  recurrent state is content: it is derived from the token stream and is
  useless without the KV that was shed alongside it. Gate the snapshot loop on
  the same `distilled` map the stream loop builds, ~25 lines above it.
- **Timeline tombstone** — already handled on apply; verify it survives the
  distill exemption, since a tombstoned-*and*-distilled timeline deliberately
  falls through the wholesale drop.

A distilled conversation becomes unresumable by construction. That is correct:
distillation is for calibration exemplars that are never resumed.

**Turn-scoped tombstones** (`tombstoned_turns`) sit awkwardly and need a
decision. A turn tombstone sheds one turn's chunks/tokens/sig while keeping the
timeline live. The snapshot is per *timeline*, not per turn, so it is not
obviously stale — but if the tombstoned turn is the one the snapshot was taken
at, its KV is gone and the conversation cannot resume there anyway. Proposal:
leave the snapshot alone, and let resume's `turn_index` validation reject it if
the recovered turn set no longer reaches that index. Flagged for review.

### 4.3 Resume

At `create_sequence`, when the slot binds to a timeline that has a snapshot:

```
loc  = conversation.recurrent_snapshot_loc(timeline)     # exists
blob = persistence.read_payload(loc)                     # exists
snap = SnapshotPayload::decode(blob)                     # exists
model.restore_recurrent(slot, snap)                      # new hook -> import()
```

`restore_recurrent` delegates to `RecurrentStateStore::import`, which already
validates `schedule_hash` + geometry and is all-or-nothing. Three rejections,
all falling back to a zero state:

- **hash mismatch** — a different model or a changed schedule. Recompute rather
  than scatter a foreign layout.
- **`turn_index` newer than the last recovered turn** — a torn shutdown.
- **no snapshot** — a conversation sealed by a model that carries no recurrent
  state, or one whose seal predates the writer. (The substrate rebuild removes
  the historical case, but the non-hybrid case is permanent.)

"Falls back to zeros" is the same silent amnesia §1 describes, so each rejection
must **log at WARN with the reason**, and the reason must be distinguishable.
A resume that quietly forgets is the failure mode this whole document exists to
remove; it must not survive as the error path.

The ordering constraint that makes this sound: `import` must run **before** the
first wave on that slot, and after the store exists. `create_sequence` is before
any `submit_turn`, so both hold.

**Interaction with the `offset == 0` reset.** A restore at `create_sequence`
leaves the map entry `Occupied`, and §5.1's survey shows the slot is back to
`offset > 0` by the time the first wave calls `ensure_recurrent` — so the reset
does not fire and the restore survives on the main path. That is correct **by
luck**, not by statement: it depends on `apply_projection` running between the
two, which nothing asserts.

The residual hole is a restored slot whose first wave genuinely does carry
`offset == 0` (an empty projection, a resume that recovers no turns). There the
reset lands on top of the restore and silently zeroes it — the exact failure
this document exists to remove. An explicit `seeded` flag on the store, cleared
by the first wave, closes it without depending on the ordering. See §9 Q2.

### 4.4 Fork: a VRAM-to-VRAM copy

Two cases, and the fast one is the common one.

**Parent live (in the model's map).** Copy the store device-to-device:

```
RecurrentStateStore::fork_from(&parent) -> Self
    per slot: DeltaNetState::snapshot()   # Tensor::copy -> try_clone, D2D
```

No host round trip, no encode, no disk. On the 35B this is ~63 MiB of device
copy per fork — sub-millisecond at PCIe-free device bandwidth.

**This cost is no longer amortised, and that is a change since the ping-pong
landed.** The original argument here was that a fork costs the same operation the
wave already ran every wave, so it was free in the sense of adding no new *kind*
of work. The wave stopped copying: it preserves the entering state by not writing
it, commit is a `mem::swap`, and the ~60 MiB per wave the store used to move is
gone. A fork's copy is now genuinely new device traffic, and §5.3a puts one on
**every dialogue turn** rather than only on an explicit `fork()`.

Two consequences to carry into P1 and P2:

- A forked slot needs **both** halves of the ping-pong: `live` is a real
  `snapshot()` of the parent's live buffer, while the write buffer can be
  allocated without initialising (the kernels fully overwrite it), so the copy is
  ~63 MiB against ~123 MiB of allocation. `advanced` starts `false`.
- Per-turn fork traffic is a number to measure, not to assume, and it is the
  first thing to look at if turn latency regresses after P2. It is also the
  reason to check whether a view's fork can be deferred to the first layer that
  actually advances, since a turn that reprojects before decoding anything has
  paid for a copy nothing read.

**Parent not resident** (daemon restart, or the parent was freed). Fall back to
§4.3's snapshot read.

Plumbing: `fork_onto` must pass the parent's sequence id through
`SchedulerRequest::NewSequence` so `create_sequence` can reach the parent's
store. Today the request carries `conversation`/`target`/`response_tx` only.
Add `parent: Option<SequenceId>`.

**The boundary constraint.** A fork's spliced KV covers the parent's history *as
sealed*. The recurrent state must correspond to the same token stream, so the
copy must be taken at a sealed boundary. If the parent has a turn in flight, its
live `S` is **ahead** of the sealed KV and copying it produces a fork whose
recurrent layers have seen tokens its attention layers have not. `Sequence`
already tracks `turn_in_flight`; fork must refuse or wait on it. This is the
fork analogue of `truncate_sequence`'s refusal to rewind to a non-zero offset,
and it deserves the same explicit error rather than a silent skew.

> This does **not** conflict with §5.3a, where a view forks the parent on every
> turn. A view is carved at the turn boundary, before any of the turn's tokens
> are decoded — exactly the sealed boundary this constraint requires. The rule
> is the same for both: fork at a boundary, never mid-turn. What differs is only
> that the view's fork is internal and automatic while a user `fork()` is not,
> so the user-facing one needs an error where the internal one needs an
> invariant.

**Join** splits in two, and only one half is hard.

*Linear join* — a child that is a continuation of its parent, reconciled back
into it — is a **move**: the child's state simply becomes the parent's, because
the child saw exactly the parent's tokens plus its own. This is not speculative
future work; §5.3a shows `finalize_view` needs it on every turn.

*Divergent join* — two children of one parent, merged — has no defined
arithmetic. `S` is an accumulated sum over one token order; there is no
operation that means "and also these other tokens". KV concatenates because it
is per-token storage; a recurrence is not. **Out of scope** until someone
states what it should mean semantically.

### 4.5 Release on free

Fix §3 defect 6 alongside: add a `release_sequence` hook to `ManagedBatchedModel`
(default no-op), have the hybrid drop the map entry, and call it from the
scheduler's free path. Then `ensure_recurrent`'s `offset == 0` rule is a
belt-and-braces second line rather than the only guard against a recycled slot
inheriting a stranger's memory.

This is worth doing **first**, before any of the above: it is a leak and
probably a correctness bug on today's code, and every path below makes slots
turn over faster.

### 4.6 The system prompt: pre-generated state per tree branch

A new conversation does not prefill its system prompt — it Arc-injects sealed
section K/V, and the wave never sees those tokens. So the recurrent state
enters the first user turn at zero while the attention layers hold the whole
prompt: the *KV-without-state* mismatch at conversation birth. Every later turn
is a fork of that state and carries forward, so this is the **only** place a
new conversation needs a state it did not compute.

It cannot be solved by caching a state per section, because state does not
compose: `v_new = u − w·Sₙᵀ` makes a chunk's contribution a function of the
state it entered with, so `state([A][B]) ≠ f(state([A]), state([B]))`. A
checkpoint is only valid for an *ordered prefix*. Fortunately the prompt
machinery is already built entirely around ordered prefixes.

**`SectionTree` already pre-seals the cross-product.** Selector nodes multiply
the possible prefixes of everything beneath them, and the tree resolves that by
sealing one [`TreeVariant`] per assignment of the dims declared above each node
— `{ ancestors: u32 (mixed-radix pack), id: SectionId, in_tree_prefix:
Vec<SectionId> }`. Changing a selector picks a pre-prefilled variant instead of
re-prefilling below it.

**So the branch key already exists.** `SectionTree::pack(selection, dim_count)`
is exactly the key a recurrent checkpoint needs: state depends on the ordered
in-tree prefix, and that is what the pack encodes. A DeltaNet checkpoint is
keyed identically to the K/V it accompanies — no new addressing scheme.

Two things have to change.

**(1) Seal the glue markers.** `TreeGlue` nodes — the dialect markers
(`<tools>`, `</tools>`) — deliberately allocate **no** sealed variants, are
prefix-transparent, and emit a `ProjectionSegment::Generated` run at projection,
re-derived every time. That is glue, inside the system prompt, and on this
lineage glue cannot be gap-filled (§5). Give them sealed variants per branch
like any mandatory node, so they are injected rather than generated. They carry
fixed content and add no dimension, so the cost is one more sealed section per
branch they are active in (`TreeGlue::active_keys` already says which), plus a
one-time re-seal of everything below them — they stop being prefix-transparent.

**(2) Checkpoint per branch, generalised over collections.** Collections are
the combinatorial problem: their members are top-k selected at runtime, so
their contribution to a prefix is not a fixed thing to key on.

The codebase has already made exactly this call for K/V, and the reasoning
transfers verbatim. `SystemPrompt::is_collection_member` excludes collection
members from the content-address prefix chain, because *"Collection members are
an approximation-rich prefix anyway — projection picks a subset at runtime, so
the section's K/V already isn't a strict function of which specific members
ingested."* Downstream sealed K/V is therefore **already** conditioned on a
prefix that ignores which members are present.

So the state should be generalised the same way, and there is a precise form
for "the same way": a placeholder node with `inject_collection` *"still seals +
emits its own content into the K/V prefix (so nodes below it anchor on a stable
placeholder, e.g. a `noop` tool)"*. **Compute the branch's DeltaNet state over
that same placeholder-substituted prefix.** Then the state and the K/V
approximate the collection identically, rather than the state carrying an
approximation of its own. That is a sharper rule than "keep a few states", and
it needs no new decision — it inherits one already made.

**What is needed at runtime is only the leaves.** Starting a conversation needs
the state *after the whole prompt* for the active assignment, i.e. one
checkpoint per full selector combination. Per-node-per-branch checkpoints are a
**build-time** accelerator — they let the generator resume the recurrence walk
from the deepest shared ancestor instead of re-walking each combination from
zero — not a runtime requirement.

**Cost.** One leaf checkpoint per combination, at ~63 MiB (§7). For scale, a
~2,000-token prompt's own sealed K/V is roughly 40 MB per variant at bf16
(2000 × 10 attention layers × 2 × 2 kv heads × 256 × 2 B), so this is the same
order as the K/V pre-generation already being paid — call it a doubling of
prompt-cache footprint, on disk rather than in VRAM. It is also the strongest
argument for §7's option 2: the dtype tag exists, and bf16 storage halves a
cost that here multiplies by the combination count rather than by the
conversation count.

**The payoff is that glue disappears from the start path.** With (1) and (2),
beginning a conversation is: inject the branch's sealed K/V, inject the
selected collection members' K/V, restore the branch's DeltaNet checkpoint,
continue. No generated runs, no gap-fill, no recompute — and §5's glue problem
stops applying to the system prompt entirely.

### 4.7 Turn boundary glue: bake it, don't regenerate it

§4.6 removes glue from the system prompt. This removes it from turns — the
other place a non-glue model meets a `Generated` run.

**A turn is a stream**, and its glue is already fully described. `StreamDecl::Turn`
carries `TurnDecl.segments: Vec<TurnSegment>`, documented as "the complete
description of the turn's K/V", with `validate_tiling` enforcing a contiguous
gap-free tiling. Glue is a first-class segment:

```rust
TurnSegment::Glue { marker: GlueKind, kv: Option<KvSpan> }
enum GlueKind { SystemStart, UserStart, AssistantStart, ImEnd, NoThink }
```

> **Naming.** `GlueKind::ImEnd` and the assembler's `assistant_end` are the same
> marker seen from two sides: the dialect string `assistant_end` (`<|im_end|>\n`)
> is what `BoundaryMarkers` tokenises, and `ImEnd` is the layout's name for the
> segment it occupies. Both spellings appear below because both appear in the
> code; they are one token sequence.

The tokens need not be stored: *"the marker text/tokens are derived from the
active dialect by this kind — the layout stores only the kind."* Kind + dialect
is deterministic and survives a dialect change, which storing bytes would not.

**Most of it is already baked.** In `TurnLayout::from_flat_grid` the intra-turn
`ImEnd` and `AssistantStart` are real (`kv: Some`), and the tool-exchange path
builds real `ImEnd`/`UserStart`/`AssistantStart` spans inside the assistant
half. Exactly three segments per turn are ethereal:

| segment | today | why |
|---|---|---|
| leading `UserStart` | ethereal | "materialized by the projection spine" |
| `NoThink` (suppressed turns) | ethereal | "live glue, not in this turn's grid" |
| trailing `ImEnd` | ethereal | same as the opener |

**And the real form is already implemented.** `from_flat_grid` bakes a real
leading boundary when `user_content_start > 0` — *"If the grid reserves room
for it, it is a baked (real) boundary"* — and the doc comment states plainly:
*"A non-zero `user_content_start` is honored as a real leading `UserStart`
boundary (**not used today, but representable**)."* The mechanism exists and is
documented as unexercised. What is missing is a caller that reserves the room.

#### The reason they were never baked

The persisted-turn contract states it directly (`substrate.rs:738-756`):

> *"the inter-turn `user_start` head and `assistant_end` tail are **not**
> persisted: the projection assembler re-emits them as live `Generated` runs at
> every cross-turn boundary **so their K vectors are computed under the actual
> runtime causal prefix**. The interior `user_end` + `assistant_start` pair
> stays baked because its semantic context … is invariant across projections."*

So the criterion is **invariance of context**, and it was applied deliberately:
interior markers are baked because their surroundings are fixed by the turn
itself; boundary markers are not, because what precedes them depends on which
turns this projection selected.

Baking them therefore *is* an approximation — the marker's K is computed under
the prefix present at seal time and reused under a different one. Three things
make it the right trade anyway:

1. **It is the approximation the substrate already makes everywhere else.** A
   sealed turn's own K/V is reused under changed prefixes on every reprojection;
   so is a sealed section's. A two-token role marker is the least
   context-sensitive thing in the projection to extend that to.
2. **The alternative is not a better answer, it is no answer.** On a lineage
   that cannot gap-fill, "recompute under the true prefix" means "refuse the
   wave."
3. **`Option` keeps the old behaviour reachable.** A `can_gap_fill: true` model
   can ignore the stored K/V and regenerate under the true prefix, exactly as
   today (see below).

What this does mean is that **the quality of a baked boundary should be
measured, not assumed** — it is the same class of question as §4.7d's
`member_glue` and belongs in the same ablation (P10.4).

#### The `Option` stays

`kv: Option<KvSpan>` should **not** become mandatory, and this is not a
compatibility concession:

- It is a genuine absent value, not a flag. `Thinking { kv: None }` uses the
  same axis for the `<think>` drain — prose kept, K/V deliberately dropped —
  and that case must survive.
- It **is** the model-capability split. A turn sealed with real boundary glue
  can be *injected* by a non-glue model **and** *regenerated* by a glue-capable
  one, which is free to ignore the stored K/V. A turn sealed ethereally can
  only be regenerated. So `Option` already means "is this boundary's K/V
  available" — a real hit/miss, which CLAUDE.md explicitly permits, rather than
  optionality-as-a-feature-flag, which it forbids.
- Forcing it mandatory would make `NoThink` unrepresentable and would bake
  boundaries for glue-capable models that gain nothing from it.

The dispatch is therefore **data, not a flag**: the assembler injects when the
K/V is there and regenerates when it is not. No dual code path, no shim.

#### What changes

1. **Reserve grid room for the leading marker.** The turn's own prefill must
   cover its `user_start` tokens so `user_content_start > 0`. This is the whole
   change at the layout level — `from_flat_grid` already does the rest.
2. **Same at the tail** for the closing `ImEnd`.
3. **`NoThink` splits in two — see §4.7b.** The sealed-turn re-render bakes;
   the live turn's switch must stay dynamic.
4. **Move the seal anchor.** `turn_start_parent_blocks` currently anchors where
   the prefix ended, i.e. *after* the leading marker. It must anchor before it,
   or the baked boundary is not in the sealed range.
5. **Stop wrapping in `assemble_pieces`.** It currently flushes `user_start`
   into the island *before* a `Sealed::Turn` and carries `assistant_end` into
   the island *after* it. With turn-owned boundaries a sealed turn emits alone.
6. **Boundary ownership — settled by the code, not a judgement call.** See
   §4.7a.
7. **A real model capability.** `ModelCoreProperties` has no "can glue" field;
   the hybrid discovers it by bailing inside `forward_wave`. The assembler
   needs to know *before* planning. This is a genuine model property like
   `head_dim`, not a feature flag.

### 4.7a Ownership is determined, not chosen

The open question was leading-owned vs trailing-owned. Reading what
`assemble_pieces` actually emits settles it — there is one arrangement that
reproduces the current token stream, and it is exact.

Per sealed turn the assembler does:

```rust
run += user_start;                       // head of THIS turn
if turn_no_think(..) { run += no_think }
flush(run);                              // island BEFORE the turn
push Turn;
run += assistant_end;                    // tail of THIS turn, carried forward
```

so the emitted stream, with `US = user_start` and `AE = assistant_end`, is:

```
[sys] │US│ TURN_0 │AE US│ TURN_1 │AE US│ TURN_2 │AE US_current│ live user …
       └island┘          └─island─┘      └─island─┘   └──island──┘
```

**Every inter-turn island is exactly `AE ++ US [++ NT]`** — the concatenation of
turn N's trailing marker and turn N+1's leading marker, in that order, because
the `AE` is appended after turn N and the `US` prepended before turn N+1 into the
same run. The `no_think` re-render, when it fires, lands *after* the `US` in that
same run, so it belongs to turn N+1 along with its opener — which is what §4.7b's
sealed-turn bake assumes. Splitting the island at that seam is therefore lossless
by construction:

| ownership | resulting stream | verdict |
|---|---|---|
| leading only (`US`) | `US body₀ US body₁ …` | **drops every `AE`** |
| trailing only (`AE`) | `body₀ AE body₁ AE …` | **drops every `US`** |
| **both (`US` … `AE`)** | `US body₀ AE US body₁ AE …` | **identical to today** |

So each turn owns its leading `UserStart` (with its `NoThink`, when present)
**and** its trailing `ImEnd`. No dialect-specific check is needed — the result is structural, and it holds for
any dialect, because the island *is* the two markers adjacent. The earlier
caution about a duplicated or dropped `<|im_end|>` resolves positively: the
seam falls precisely where ownership changes hands.

### 4.7b `NoThink` is two different things

Change 3 above originally said "bake `NoThink`". That is wrong as stated,
because two distinct segments carry that marker and only one of them may be
baked.

**The sealed-turn re-render** (`assemble_pieces`, guarded by
`turn_no_think(rt.timeline, rt.index())`) exists so a suppressed turn stays
self-consistent with its own empty `<think></think>` — *"instead of an
unexplained empty block the model learns to mimic."* It is keyed to that turn's
own persisted flag, which is fixed once the turn is sealed. **Safe to bake**,
and it must be, or a non-glue model cannot reproduce it.

**The live turn's switch** (`scheduler/mod.rs:2833`) is explicitly the
opposite: *"re-decided from the current dial every projection, never sealed
into a turn — keeps it out of the substrate and prevents a past suppressed turn
from leaking a stale switch onto a later thinking-on turn."* Baking that would
freeze a deliberately dynamic decision. **Must not be baked.**

### 4.7c The tail is prefill, not glue — and the last island can vanish

A distinction the earlier sections blurred, and it decides how much of this
matters:

> **Gap-fill is filling a hole in the middle of a sequence. Appending at the
> writer tail is ordinary prefill.** The hybrid bails on `n_glue > 0` — glue
> *rows*, i.e. reserved gap chunks — not on prefill rows.

Today the live turn's opener is a separate `Generated` segment
(`user_start_current`), which `assemble_pieces` flushes into a trailing island,
and `reserve_glue_island` reserves a gap chunk for it like any other — there is
no special case for a trailing island. So the live turn currently costs a glue
island it does not need.

Under §4.7a the `AE` moves into turn N's tail, shrinking that island to
`US_current [+ no_think]`. It can be removed entirely by **prepending those
tokens to the `NewUserMessage` tokens** instead of emitting them as a separate
`Generated` segment. Then:

- there is no trailing island at all — the live turn is ordinary tail prefill;
- the live `no_think` stays dynamic (the deferred user tokens are rebuilt every
  projection anyway), satisfying §4.7b without a second mechanism.

This is the change that makes the turn stream glue-free end to end, and it is
smaller than the ones around it.

### 4.7d `member_glue` and template sections

Both remaining `Generated` producers are the same shape and get the same
answer.

`SectionCollection::member_glue` is a separator emitted *between* consecutive
selected members — genuinely interstitial, so genuinely gap-fill. It is unbaked
on purpose, *"so it is independent of which members provenance selects."*
`is_template` sections are the non-tree twin of §4.6(1)'s `TreeGlue`: dialect
structural text, live-prefilled so it *"stays attention-correct under whatever
prefix the projection"* produces.

Both are unbaked for the same reason — their prefix varies with selection — and
that reason is already conceded elsewhere: collection members are deliberately
excluded from the content-address chain because *"the section's K/V already
isn't a strict function of which specific members ingested."*

So bake both **against a canonical prefix**, accepting the approximation the
collection path already accepts. For `member_glue` specifically, bake it as a
**leading** separator on every member: the spurious copy then lands at the head
of the collection block, which is already a structural boundary, rather than
against whatever section follows the last member.

The decisive argument is that on a non-glue model the alternative to an
approximate bake is not a more accurate answer — it is a wave that refuses to
run.

#### The glue cache stops being a cache

`SlotState.glue_islands` is in-memory, per-slot, keyed by content-context hash,
retained `GLUE_ISLAND_RETAIN_GENERATIONS = 4`. A fresh slot — fork, restart, new
conversation — starts empty, so every island misses and must be computed. That
is exactly where a non-glue model fails, and it is why the existing cache cannot
be the answer.

With turn-owned boundaries there is **nothing to cache**: the glue K/V is part
of the turn's sealed chunks, persisted by the existing `Chunk` records and
restored by the existing resume path. No new record type, no new index, no
lifetime question.

Once every producer below is resolved, `SlotState.glue_islands` has nothing to
hold on this lineage. It stays in the tree for glue-capable models, which may
still choose to regenerate rather than inject (§4.7's `Option` is a hit/miss,
not a mode) — but for the hybrid it becomes dead weight, and should be treated
as such rather than left as an apparently-live cache that never fills.

#### What still generates, after §4.6 and §4.7

Five producers of `ProjectionSegment::Generated` exist:

| producer | after these changes |
|---|---|
| `TreeGlue` dialect markers (`project.rs:1905`) | sealed — §4.6(1) |
| turn boundary markers (the spine) | sealed — this section |
| `is_template` sections (`project.rs:1988`) | sealed against a canonical prefix — §4.7d |
| `SectionCollection::member_glue` (`project.rs:2095`) | sealed as a leading separator — §4.7d |
| the live turn's opener (`user_start_current`) | folded into `NewUserMessage` — §4.7c |

With all five resolved, a projection of system prompt + sealed turns + a live
user message emits **zero** gap-filled glue. That is the condition for running
this lineage at all, and it is now met by construction rather than by a
fallback.

### 4.8 Provenance signatures: derive the fold, don't hardcode it

Provenance captures each token's `sign(Q)` and scores a live decode window
against past turns' signatures. On the hybrid it is attention-only — three
quarters of the stack has no Q in a KV cache to read — and four constants tuned
for a 48-layer, 4-KV-head, head_dim-128 stack silently misbehave.

#### How capture works, and why compression does not threaten it

`R16` is the **active** K format for every live GPU sequence: raw F16 *plus
reserved Q-capture space*, twice plain F16. The C0–C10 ladder is the *sealed*
format. The seal gathers signatures **before** quantizing, and says so:

> *"Capture the whole turn's wide per-token `sign(Q)` from R16 NOW — before
> `record_turn` detaches the sealed KV… a block whose R16 is already gone
> (compressed) simply contributes nothing."*

So compression needs no provenance exemption, and there is no per-layer
compression policy to give one. The consequence to hold onto is that **capture
is one-shot**: once a block leaves R16 the Q is gone, and a signature can only
be re-captured by re-prefilling — which is exactly what the `<think>` re-prefill
path does deliberately, *"so they match the sealed K/V."*

#### What breaks, and what already works

Already correct: `qwen35::engine::provenance_layer_indices` snaps each depth
band down to an attention layer and makes its lower endpoint the *previous
attention layer*, so the bands span layers that actually have signatures. That
work is done. (Those bands feed the `kv-zero-check` diagnostic; the sig path
itself passes every KV layer.)

| # | constant | today | fails on the hybrid because |
|---|---|---|---|
| 1 | `PROV_HEADS_PER_LAYER` (= `HEADS_PER_GROUP`) | `4` | the 35B has **2** KV heads, so `n_layers = n_heads / 4` reads 10 layers × 2 heads as 5 × 4 — structure scrambled, not merely mis-sized |
| 2 | `PROV_FOLD_SIZES` | `[46, 1, 1]` | a 48-layer shape; at ≤10 layers group 0 absorbs everything and **groups 1 and 2 are empty** — ⅔ zeros, and the identity-bearing top layers vanish |
| 3 | `PROV_FOLD_SHIFT` / `rotate_head` | `32`, gated on `wph == 2` | head_dim 256 ⇒ `wph == 4` ⇒ the decorrelation stagger **silently no-ops**, so correlated layers cancel dim-aligned |
| 4 | `n_palette` | `4` | `prov_sub_head_dim()` returns 0 when `head_dim/4 > 32`, so at 256 the GPU sign-pack path is **skipped** and every seal pulls the whole R16 dump D2H |

Item 4 is a prerequisite rather than part of the fold, and it is the dangerous
one: it is not a correctness bug, so it presents as "provenance got slow on the
new model" rather than as a disabled fast path. It plugs into the same
per-backing `n_palette()` mechanism §4.2 of `qwen35_qwen38_models.md` already
scopes for the decode kernels.

It also cannot be fixed in `prov_sub_head_dim` alone. Three lines below that
call, the same function stamps `n_palette: candle_nn::kv_cache::N_PALETTE` — the
**global constant** — into the `ProvSignPacked` it returns, beside the
`sub_head_dim` it just derived
([batched_inference.rs:1924-1936](../candle-transformers/src/models/batched_inference.rs#L1924-L1936)).
Deriving one and not the other does not leave the old behaviour in place; it puts
two mutually contradictory numbers in one struct, so a consumer reconstructing
`head_dim = n_palette × sub_head_dim` reads 128 for a 256-dim head. Both move
together or neither does.

#### The generalisation

Each constant becomes a value derived from the model, and **every derivation is
an identity on Qwen3-30B**:

| constant | derivation | at Qwen3-30B |
|---|---|---|
| heads per group | the model's `n_kv_head` | 4 ✓ |
| fold sizes | `[n − 2, 1, 1]` over capture layers | `[46,1,1]` ✓ |
| fold shift | `head_dim / 4` | 32 ✓ |
| `n_palette` | smallest `p` with `head_dim / p ≤ 32` | 4 ✓ |

`[46,1,1]` turning out to be exactly `[n−2,1,1]` is the useful one: the existing
constant already *is* the general rule, written out longhand.

Two need real code rather than parameterisation:

- **`rotate_head`** does a `u128::rotate_left` and bails unless `wph == 2`.
  Generalise to a word-wise rotate over `wph` words, with a bit-identity test at
  `wph == 2`.

  > **Generalising `rotate_head` alone is a no-op, and a silent one.** Two things
  > block it, and neither is inside the function. Its signature returns
  > `(u64, u64)` — a shape that cannot carry a `wph == 4` result at all — and
  > `fold_provenance` guards the *use* of that result with a second
  > `if wph == 2 { xor rotated } else { xor raw }`
  > ([wide_sig.rs:124-131](../candle-conversation/src/provenance/wide_sig.rs#L124-L131)),
  > so at head_dim 256 the caller XORs the unrotated words no matter what the
  > rotate does. A change that fixes only the function compiles, passes a test
  > written against the function, and leaves production folding without a
  > stagger. Both the return type and the caller branch are part of the change
  > (P5e.4, P5e.4a).
- **`HEADS_PER_GROUP`** is aliased into `scan.rs`, `gpu.rs` and `packed.rs`. The
  scorer must use the value the signature was **folded with**, not a
  compile-time constant — which is why it has to ride on the record (below).

  There is a **fourth** copy, and it is not an alias:
  [zend/examples/provenance_layers.rs:34](../zend/examples/provenance_layers.rs#L34)
  declares its own `const HEADS_PER_GROUP: usize = 4`, so a grep for
  `PROV_HEADS_PER_LAYER` does not find it. Its `project_groups` slices the
  signature at `g × HEADS_PER_GROUP × wph` word strides; on the hybrid's folded
  shape (6 heads × 4 words) that stride is twice the real group width, the
  bounds check `e <= words.len()` fails, and the function returns an **empty**
  projection — which the scorer then ranks as though it were valid. This is a
  layer-attribution harness whose entire output is which layers carry the
  signal, so a silent empty projection is the one answer it must not give.

#### Compatibility is a format contract, not a code path

Two requirements, neither of which is a shim:

1. Qwen3-30B must fold **bit-identically**, so existing `WideQSig` galleries stay
   valid and today's measured retrieval numbers still hold.
2. **A signature is only comparable to another under the same fold.** Mixing
   folds does not degrade gracefully — Hamming distance between differently
   folded bit vectors is meaningless, and the scorer will return confident
   nonsense.

So: stamp the fold parameters onto the signature record, and **refuse to score
across a mismatch**, with a distinguishable warning. Making an implicit format
assumption explicit is not backward compatibility; it is the thing that lets the
derivation change safely at all.

#### The budget does not shrink

"10 layers instead of 48" sounds lossy. It is not, in bits:

```
Qwen3-30B:  3 groups × 4 kv-heads × 128 bits = 1536 bits
hybrid:     3 groups × 2 kv-heads × 256 bits = 1536 bits
```

Fewer heads, twice as wide. What changes is **fan-in**: a group head XORs 46
layers' sign bits today and 8 on the hybrid. Less XOR mixing means less
cancellation, which cuts *against* the assumption that fewer layers is worse.

#### Does attention-only deliver the spirit?

Probably, and the argument is structural rather than hopeful: this architecture's
premise is *provenance-selected attention* — retrieval happens in the attention
layers, over a retrieved subset. On a 3:1 hybrid those 10 layers are the
retrieval layers, and the 30 recurrent layers do fast local mixing over a
compressed state. Capturing `sign(Q)` from exactly the layers that perform
retrieval is better targeted than capturing uniformly across a stack where every
layer attends.

The argument against is worth stating rather than burying. `[46,1,1]` encodes a
**measured** finding — identity lives in the top two layers. On the hybrid, "the
top two attention layers" are **35 and 39**: four apart, with three recurrent
layers transforming the residual between the two captures. That could sharpen
the signal or scatter it, and reading the code cannot say which.

So `[n−2,1,1]` is the **default**, not the answer. Treat the fold shape as a
per-model constant to be re-derived by measurement — the same rule CLAUDE.md
already applies to the quantization thresholds ("must be re-derived for each
variant"). That measurement is the Phase 4 exit criterion risk 5 already asks
for, so it is scheduled work, not new work.

#### Guards

- **Refuse on fold mismatch**, distinguishably. A gallery scored under the wrong
  fold returns plausible numbers, which is the worst failure this subsystem can
  produce.
- **Never emit an all-zero group.** `[46,1,1]` on a short stack does exactly
  that today. The fold should refuse to build a signature it cannot fill, the way
  `provenance_layer_indices` already returns `None` for a stack with no
  attention layers.

## 5. Truncation: two operations under one name

### 5.1 The survey

There are **two** truncation primitives, not one, and conflating them is what
makes a rewind look easier to delete than it is:

- `truncate_sequence_to_blocks(slot, n_blocks)` — the chunk-granular form, on
  both `ChunkedKvBacking` and `BatchedInferenceSession`.
- `truncate_sequence_to_tokens(slot, n_tokens)` — the token-granular form. **This
  is the one that expresses a rewind.** `ManagedBatchedModel::truncate_sequence`
  bottoms out here ([batched_inference.rs:3258](../candle-transformers/src/models/batched_inference.rs#L3258)),
  and so do both of its overrides — the hybrid's
  ([qwen35/forward.rs:190](../candle-transformers/src/models/qwen35/forward.rs#L190))
  and deepseek4's
  ([latent_moe/wave.rs:916](../candle-transformers/src/models/latent_moe/wave.rs#L916)).

The `_to_blocks` sites, sorted by *target* rather than by caller, separate
cleanly:

| # | site | target | what it is actually for |
|---|---|---|---|
| 1 | [mod.rs:4646](../candle-conversation/src/scheduler/mod.rs#L4646) | `0` | post clean-seal: drop the slot's `Arc<ChunkGid>` refs, residence owns them |
| 2 | [mod.rs:5622](../candle-conversation/src/scheduler/mod.rs#L5622) | `0` | post-`Done` seal: identical housekeeping |
| 3 | [mod.rs:5731](../candle-conversation/src/scheduler/mod.rs#L5731) | `0` | `prepare_section_ingest` — "defensive truncate" before injecting a prefix |
| 4 | [mod.rs:7551](../candle-conversation/src/scheduler/mod.rs#L7551) | `0` | free a view, clear the parent so `apply_projection`'s populated-slot guard lets the rebuild through |
| 5 | [projection_assembler.rs:625](../candle-conversation/src/scheduler/projection_assembler.rs#L625) | `0` | `apply_segments_build` — snapshot tail, clear, rebuild in logical order |
| 6 | [prefill.rs:2475](../candle-conversation/src/scheduler/prefill.rs#L2475) | `0` | error path: reprefill failed, drop the slot's chunks |
| 7 | [batched_inference.rs:1399](../candle-transformers/src/models/batched_inference.rs#L1399) | `0` | seal quantize: drop the float chunks, `inject_sealed_at_tail` the quantized ones |
| 8 | [mod.rs:5490](../candle-conversation/src/scheduler/mod.rs#L5490) | **`seal_block_from`** | `<think>` clean-reprefill: rewind to the turn boundary |
| 9 | [batched_inference.rs:1211](../candle-transformers/src/models/batched_inference.rs#L1211) | **`pre_counts[li]`** | `reserve_glue_gap` rollback: undo a partially-reserved gap chunk |

Plus the `_to_tokens` sites, which the block-count view does not see at all:

| # | site | target | what it is actually for |
|---|---|---|---|
| 10 | [batched_inference.rs:3482](../candle-transformers/src/models/batched_inference.rs#L3482) → `:3258` | **`poss[i] + kept`** | speculative-decode accept: rewind to the accepted prefix |
| 11 | [alloc.rs:1722](../candle-nn/src/kv_cache/chunked/alloc.rs#L1722) | **`target_tokens`** | `heal_tail_divergence`: trim a failed wave's surplus token off every layer |
| 12 | [cache.rs:714](../candle-nn/src/kv_cache/cache.rs#L714) | any | `truncate_chunked_to_tokens` — the `KvCache` passthrough |
| 13 | [batch_test/utils.rs:580](../candle-transformers/src/models/batch_test/utils.rs#L580) | **`prompt_len`** | `decode_replay_probe`: roll back between replays |
| 14 | [batch_test/utils.rs:1182](../candle-transformers/src/models/batch_test/utils.rs#L1182) | **`base`** | the gate harness's repeat loop: re-prefill "from identical state" |

**Sites 1–7 are not rewinds.** Every one targets zero, and every one is either
immediately followed by a repopulation in the same operation or is the last
thing that happens to a dying slot. Their meaning is *release these chunk
references* / *clear so I can rebuild*, and the slot's logical content either
does not change or is rewritten wholesale. They are safe for a recurrent model
exactly as they stand — and they are safe **by accident**, because the slot is
back to `offset > 0` before the next wave calls `ensure_recurrent`, so the
`offset == 0` reset never fires on a conversation that is merely being rebuilt.

**Sites 8–14 are genuine rewinds** — "put this sequence back the way it was `n`
tokens ago" — and that is an operation a recurrent state **cannot express**.
`S` is an accumulated sum with no per-token decomposition. There is no inverse.

Only site 10 routes through `ManagedBatchedModel::truncate_sequence`, which is
why the hybrid's `tokens != 0` bail exists — and why every other rewind, which
reaches the session or the backing directly, slips past it.

Three of them deserve naming individually, because they are not merely unguarded:

- **Site 9** is the reason the block-count parameter cannot simply be deleted.
  It is an error-path rollback of a partially-reserved glue gap, and its target
  is the pre-reservation chunk index — non-zero whenever the slot holds a prefix,
  which for a glue gap it always does. It is inside `reserve_glue_gap`, so on a
  `can_gap_fill: false` model it never runs; that is a defensible answer, but it
  has to be stated rather than assumed, and P3 has to rewrite the call.
- **Site 11** trims a failed wave's surplus token off every layer. `rollback_wave`
  should already have restored `S` to the wave-entry value, so KV and state
  probably agree afterwards — but "probably" is doing real work in that sentence,
  and §5.3a's disposition table does not cover a path that *trims* a slot rather
  than clearing it. It needs a test, not a reading (T3.7).
- **Sites 13 and 14 are the gate harness, and they are already broken for this
  lineage.** `decode_replay_probe`'s own comment says it replays through
  `truncate_sequence_to_tokens` precisely so that *"if that did not restore
  byte-identical state the probe would report its own drift as model
  non-determinism."* On a hybrid it does not restore it: the probe's wave
  committed, `S` advanced by a token, and rolling the KV back cannot reach it.
  So each replay enters with a further-advanced state and the control reports
  harness drift as model drift — the instrument-lies failure mode, on the
  instrument used to rule that failure mode out. Site 14 is the same defect in
  the repeat loop of `test_parallel_batched_forwarding`, whose comment likewise
  claims "every repeat is a true re-prefill from identical state."

That last pair is why P3 cannot be scoped to the `_to_blocks` family: these are
the two sites that fire during the hybrid's own model gates, and no assertion
about block counts can see them.

### 5.2 The replacement: fork instead of rewind

The proposal is to **remove the ability to rewind at all**, and express every
rewind as *keep the parent, fork, discard the child*. A fork is a device-to-
device copy of the state buffers (§4.4); discarding a child is free. So the
operation that has no inverse is replaced by one that never needs an inverse:
we simply never advance the state we want to keep.

That collapses the API to a single safe primitive — but it has to be applied to
**both** families, or the rewind survives in the one the rename does not touch:

- `truncate_to_blocks(slot, 0)` → renamed to what it does, e.g.
  `release_slot_chunks` / `clear_slot`. Valid for any model, no recurrent
  interaction, no guard needed.
- `truncate_to_blocks(slot, n > 0)` → **deleted**. Only site 9 uses it, and it
  becomes an explicit "release the chunks reserved by this failed call."
- `truncate_to_tokens(slot, n)` → **deleted outright.** There is no zero-target
  survivor here to rename around: every caller is a rewind. Sites 8 and 10 become
  forks; sites 13/14 become fresh sequences; site 11 becomes a discard; site 12
  goes with its only caller.

The win is that the dangerous operation stops being *guarded* and starts being
*inexpressible*. The hybrid's `tokens != 0` bail can then go away, because
there is no longer an API call that means it — **which is only true if the token
form goes too.** A phase-3 gate stated in terms of block counts would pass green
with `truncate_sequence_to_tokens` and all four of its non-zero callers intact.

### 5.3 Site 8 — the `<think>` clean re-prefill

Today: decode the response (which advances `S` through every DeltaNet layer,
thinking tokens included) → truncate KV back to the turn boundary → re-prefill
the turn reasoning-free (advancing `S` **again**, on top of a state that already
absorbed the thinking). The state ends up having seen
`[prefix][thinking response][clean response]` while the KV holds
`[prefix][clean response]`.

This is not a fork or resume problem: it is every dialogue turn that thinks, on
a conversation that was never forked or resumed, and it compounds. It is latent
only because zend still runs Qwen3-30B.

**Decision: fork at turn start (option (a)).** The parent stays frozen at the
turn boundary and the turn decodes on a child; sealing clean means discarding
the child and re-prefilling onto the parent, whose `S` then advances over
`[prompt][clean response]` exactly once. The rejected alternative was to
restore from the turn-boundary snapshot (§4.1) — it reuses machinery we are
building anyway, but it repairs correctness after the fact rather than making
it structural, and it needs ~63 MiB resident per live conversation (1.3 GB at
×20) to avoid a D2H/H2D per rewind.

### 5.3a Fork-at-turn-start already exists: it is called a *view*

This is the discovery that reframes the work. The scheduler **already** carves
a child slot at turn start and decodes on it:

```
submit turn  → create_view(parent, ranges) → a NEW sequence id borrowing the
               parent's KV blocks zero-copy      (mod.rs:6699-6724, 2988)
decode       → runs on the view
reproject    → free the view, rebuild the parent, carve a FRESH view   (7551)
seal         → cleanup_finished seals parent[turn_start_parent_blocks..end]
finalize     → the view's decoded blocks transfer to the parent
```

`create_view_sequence` calls `create_sequence()`, so a view is a genuinely
distinct sequence id, not an alias ([batched_inference.rs:1533-1551](../candle-transformers/src/models/batched_inference.rs#L1533-L1551)).
The parent/child split option (a) describes is the existing turn machinery.
Nothing new needs building — the slot budget already accommodates parent +
view per turn, and the "discard the child" path is already exercised on every
reprojection.

**And that is also a much worse bug than §3 defect 7.** The recurrent state is keyed
by sequence id. A view has its own id, so `ensure_recurrent` takes the
`Vacant` arm and hands it **zeros** — while its KV borrows the parent's entire
history. The view then decodes the whole turn with no recurrent memory at all,
`finalize_view` transfers only its KV blocks back, and the view's state is
dropped on the floor.

So on a hybrid model the recurrent layers never carry anything across a turn
boundary in *either* direction: the parent's `S` is never advanced by a turn's
decode, and each turn's view starts from zero. Three quarters of the stack
contributes nothing but a fixed function of the current turn, and the model
runs on its ten attention layers. It would read fluently and be structurally
lobotomised — the §7.8 signature exactly.

This fires on every dialogue turn on today's code. It is not a fork feature
gap; it is the reason the hybrid cannot be switched on as-is.

**What the fix is.** Views need two things the KV path already has:

- **carve** — `create_view` device-copies the parent's `RecurrentStateStore`
  into the view's slot (§4.4's `fork_from`), rather than leaving it vacant.
- **finalize** — `finalize_view` moves the view's advanced state to the
  parent, the recurrent twin of transferring its decoded blocks. This is a
  *move*, not a merge: a view is a linear continuation of its parent, so the
  child's state simply becomes the parent's.

That last point is where §4.4's split of *join* comes from. Join along a linear
continuation is a move, and it is not future work — the turn loop needs it on
every turn. Only a divergent join stays out of scope.

#### The disposition rule

A view ends in one of two ways, and **which one is per-transition, not per
view.** This is the sharpest thing in the section and it was previously stated
only implicitly at each site:

| transition | KV today | recurrent state under the fix |
|---|---|---|
| mid-turn reprojection | `finalize_view` transfers the decoded blocks to the parent, then a fresh view is carved | **move** — the turn's decode so far is real and must survive |
| clean-reprefill seal (`<think>`) | the decoded blocks are abandoned; the turn is re-prefilled clean | **discard** — those tokens are being un-said |
| ordinary seal | blocks transfer, turn sealed from the parent | **move** |
| wave failure / error path | slot cleared | **discard** |

The rule is simply: **the recurrent state follows the K/V.** Wherever
`finalize_view` transfers blocks, the state moves with them; wherever the blocks
are abandoned, the state is dropped with them. Getting this backwards is silent
in both directions — a discard where a move belonged loses a turn's decode, and
a move where a discard belonged reintroduces exactly the `<think>` skew this
work removes.

**And that rule is the whole `<think>` fix.** Today's flow rewinds the parent
because the thinking tokens landed on it. Once the view carries the state, the
clean-reprefill path simply *does not finalize*: discard the view, re-prefill
clean onto the untouched parent. Site 8's truncate disappears because there is
nothing to rewind — the outcome §5.2 is aiming at.

It also settles the §1 table's reprojection row. Reprojection carries the state
forward **because it finalizes**, not because reprojection is special.

### 5.4 Site 10 — speculative decode

Today this is not a skew but a hard stop: the hybrid's bail makes the first
accept with `kept > 0` fail, so **speculative decode is structurally
unavailable on the whole hybrid lineage**. The bail reads like a defensive
check; it is actually a capability block.

Forks give it a defined answer for the first time. Fork before the verify
block; decode the block on the child; on accepting `k` of `n`, discard the
child and advance a fresh fork of the parent over the `k` accepted tokens as a
single batched prefill. The cost is `k` tokens of re-advance against a block of
`n` that was just verified — cheap, and it is one batched prefill rather than
`k` decode steps.

This is not on the critical path for the switch, but it should be recorded:
it is the reason spec-decode is off, and it stops being a dead end once forks
exist.

### 5.5 Sites 1–7

No behaviour change. The work is renaming, so that a future reader cannot
reach for "truncate" and get a rewind, plus a comment at each site stating that
the slot is repopulated before the next wave — which is *why* the recurrent
state is untouched, and is currently true by luck rather than by statement.

Site 6 (error path) deserves a moment at review: the reprefill failed, the turn
is lost, and the slot is cleared. Under fork-based rewind the answer is simply
to discard the child, which is strictly better than today.

## 6. Test strategy

Three tiers, because the failure modes here are the kind that pass every
structural check. §7.8 of `qwen35_qwen38_models.md` is the standing warning: all
three defects that cost the hybrid its bring-up left the model fluent, shape-
correct and error-free. Nothing in this document can be validated by "it ran."

| tier | runs | needs | gate |
|---|---|---|---|
| **1 — fast** | every `cargo test`, seconds | CPU only, no model, no GPU | every merge |
| **2 — scenario** | `--ignored`, explicit | GPU and/or a real checkpoint | before each phase lands |
| **3 — zend** | `--ignored`, explicit | daemon + real model + substrate | before the switch |

The ordering principle is that **tier 1 must be able to fail for every defect
in §3.** A bug that only tier 3 can catch is a bug found after a 4-minute model
load, in a run that also exercises fifty unrelated things — which is how the
original three stayed hidden.

### 6.1 What already exists

More than expected. These are harnesses to extend, not to build:

| area | existing tests | file |
|---|---|---|
| recurrent store | export/import round-trip, hash + geometry refusal, wave rollback/commit, **`a_wave_leaves_the_entering_buffer_untouched`** (the ping-pong contract — this replaced the old write-back-into-the-live-buffer test, which pinned the inverse), wave sequencing, `schedule_hash` pins layout | `delta_net/state_store.rs` (6) |
| snapshot codec | round-trip, exact header bytes, unknown-version rejection | `persistence/record.rs` |
| snapshot lifecycle | single tail survives reload + relocation, exact dead-byte credit, superseded not relocated, tombstoned not relocated | `persistence/maintenance.rs` (4) |
| turn layout | tiling + realisation, ethereal segments add no K/V, **`leading_boundary_offsets_user_body`**, thinking split, tool-exchange split, tiling rejects gaps | `turn_layout.rs` (10) |
| assembler | `assemble_pieces_wraps_turns_merges_glue_and_defers_user`, no-think re-injection, consecutive-generated coalescing, glue bridge windows | `scheduler/projection_assembler.rs` (6) |
| provenance | fold groups layers with stagger, `from_band` packs signs, history round-trip, exact header bytes, late-fusion vote, per-group isolation | `provenance/wide_sig.rs` + `scan.rs` (9) |
| mixer | chunked ≡ sequential at every chunk width, segmented ≡ one-shot, GQA tiling, strong-decay finiteness | `delta_net/mix.rs` (12) |

Two of these are load-bearing for the new work and worth naming:

- **`leading_boundary_offsets_user_body`** already covers §4.7's change 1 — a
  real leading `UserStart` with `user_content_start > 0`. The mechanism is
  tested; only the caller is missing.
- **`assemble_pieces_wraps_turns_merges_glue_and_defers_user`** is the exact
  test §4.7a's stream-equivalence assertion has to preserve. It encodes today's
  `AE ++ US` island shape, so it is both the thing to keep green and the source
  of the expected stream.

### 6.2 Tier 1 — fast, CPU, no model

**The enabler already exists.** `scheduler/mod.rs` has a `DummyModel`
implementing `ManagedBatchedModel` over CPU tensors, plus `make_test_scheduler()`
which stands up a real `Scheduler` on `Device::Cpu` with a dummy tokenizer and
tiny arenas. Scheduler-level behaviour is testable with no GPU and no
checkpoint.

**The one piece to build is a `DummyRecurrentModel`** — `DummyModel` plus the
three new hooks (`export_recurrent`, `restore_recurrent`, `release_sequence`)
over a toy state (say 2 layers × 2×2, F32). That is the whole tier-1
infrastructure cost, and it buys something better than a real model would: with
a state that small, every assertion is an **exact integer comparison**, not a
tolerance. A state that should be `[[1,2],[3,4]]` either is or is not.

Everything in §6.4 below marked *fast* runs here. In particular all of:

- the codec / store / fold / layout / assembler groups (pure functions);
- **the view disposition rule** (§5.3a) — move on reprojection, discard on
  clean-reprefill — driven through the scheduler against `DummyRecurrentModel`.
  This is the correctness core of the whole document and it needs no GPU;
- the recycled-slot defect (§3 defect 6): free a slot, allocate again, assert
  the new sequence does not inherit the old state;
- fork: parent/child bit-identity **and distinct allocations** (mutate the
  child, re-read the parent — the test that catches a `Clone` sharing storage);
- zero-gap-chunk projection (§4.7c/d), asserted on reserved gap count.

Nothing here needs the 35B, and nothing here should be allowed to become slow:
if a tier-1 test starts needing a model, it belongs in tier 2.

### 6.3 Tier 2 — explicit scenario tests

`#[ignore = "…"]` with a reason that states the cost and how to run it, matching
the repo's existing convention (`"downloads the pinned Qwen3.5-9B GGUF (7.5 GB)
and needs a GPU. Run with: …"`).

These cover what tier 1 structurally cannot — real weights, real geometry, real
kernels:

- **Real-geometry fold** (§4.8): the derived parameters against the 35B's actual
  10 attention layers / 2 KV heads / head_dim 256, plus the Qwen3-30B
  bit-identity check. Needs a checkpoint because the geometry comes from it.
- **Seal → drop → resume** on a real model, asserting the restored state is
  **bit-identical** to the exported one. Not a tolerance: it is a byte copy, and
  a tolerance would hide a layout bug.
- **Parent/fork continuation**: identical continuations from the fork point
  under identical sampling.
- **The `<think>` oracle**: N thinking turns produce the same sealed state as
  the same turns with thinking disabled.
- **`n_palette` path assertion** (§4.8): at head_dim 256 the GPU sign-pack path
  is *taken*. Assert the path, not the timing — the fallback is correct, so only
  a path assertion catches the silent demotion.
- The existing GPU-gated mixer/kernel parity tests, unchanged.

The 9B is the right default checkpoint for these: it is 7.5 GB rather than 22,
and it is 16 QK / 32 V heads, so the GQA broadcast is live (§7.8 defect 3 is
invisible on the 0.8B, where `h_v == h_k`).

### 6.4 Tier 3 — zend

`zend/tests/` already has the shape: `#[ignore]`d integration tests with tiered
reasons ("Tier 3 (cruise): ~200 turns, ~30 min"; "Tier 3 (stress): ~2 000 turns,
~5 h weekly"). The hybrid needs three additions, and they should be the *last*
gate rather than the first:

1. **Ingest** — a workspace ingests and seals under the hybrid, with snapshots
   written, compaction keeping exactly one tail per conversation, and distilled
   timelines shedding theirs.
2. **Conversation continuity** — the `infinite_conversation_smoke` shape, re-run
   against the hybrid: turns accumulate, recall works, and the recurrent state
   is non-zero at depth (a state that has silently zeroed reads as fluent
   amnesia, so assert the state, not just the text).
3. **Restart** — stop the daemon, restart, resume the timeline, and continue the
   same conversation coherently. This is the only tier that exercises
   `fork_resuming` against a real redo log.

Tier 3 is where "it ingests and runs conversations" is confirmed. It is not
where any of this is *debugged* — every defect in §3 has a tier-1 or tier-2 test
above that fails first and localises better.

### 6.4a The behavioural catalogue

The tiers above describe tests scoped to *this* implementation. A second
catalogue — `recurrent_state_behavioural_tests.md` — states the same system as
invariants over observable behaviour, deliberately without reference to how any
of it works, so that the tests can be written before the code that satisfies
them and can outlive a rewrite of it.

That framing has already earned its place: two of its entries could not be
written without answering a design question first, and one of those (parallel
scope ingest splicing turns onto a conversation whose memory never saw them) is
a **live gap** on the `code_read` path that every implementation-scoped test in
this document walks straight past.

### 6.5 The oracles that do not need a reference model

Worth stating separately, because they are what actually settled the last
round of hybrid defects and they are cheap:

- **one-shot vs segmented** — run N turns straight through, then the same N with
  a seal/drop/resume (or a fork, or a reprojection) in the middle, and compare.
  Needs no reference implementation, which is what makes it decisive when the
  reference is itself suspect. `wave_one_shot_equals_segmented` already exists
  in `qwen35/forward.rs`; the state-persistence analogue is the same shape.
- **paired disposition** (§5.3a) — move-on-reproject and discard-on-reprefill
  must be asserted *together*; either alone passes under a naive "always move"
  or "always discard" implementation.
- **a factual one-liner on the real checkpoint** — "does this model still know
  Paris" separates a mis-read checkpoint from a model that cannot follow an
  instruction, and those look identical in a gate diff.

### 6.6 Per-change coverage

The concrete assertions, per change. Built alongside the code, not after.
Unmarked entries are **tier 1** (fast, CPU); tier 2 and tier 3 are called out.

**Codec / store (CPU, no GPU).**
- `SnapshotPayload` round-trip at real 35B geometry — already exists for tiny
  dims; extend to 30 layers × 32 heads × 128².
- `import` refuses a wrong hash, a wrong layer count, a wrong per-layer
  geometry, and a short blob, leaving the store untouched in every case.
- Export → import → export is byte-identical.

**Persistence.**
- Two snapshots for one timeline: the second supersedes, compaction keeps one,
  and the surviving bytes are the *second*.
- A distilled timeline's snapshot is dropped by compaction; its `WideQSig`
  survives. (The inverse of the existing distillation test.)
- A tombstoned timeline's snapshot is dropped.
- Plan/execute race: a snapshot appended between plan and execute is the one
  that survives, and the planned copy is not relocated. (Extends the existing
  `live_snapshots` tests to the new writer.)

**Resume (the decisive one).** Tier 1 against `DummyRecurrentModel`; the
real-checkpoint repeat is tier 2.
- Seal a turn, drop the store, resume from the log, and assert the restored
  state is **bit-identical** to the exported one. Not a tolerance — this is a
  byte copy, and a tolerance here would hide a layout bug.
- Resume with a mismatched `schedule_hash` → zeros **and** a WARN.
- Resume where `turn_index` exceeds the recovered turns → zeros and a WARN.

**Fork.**
- Fork a live parent; assert the child's `S` and conv tail are bit-identical to
  the parent's and that the buffers are **distinct allocations** (mutate the
  child, re-read the parent). This is the test that catches a `Clone` that
  shares storage — the exact hazard `DeltaNetState`'s missing `Clone` impl
  exists to prevent.
- Fork with a turn in flight → explicit error, not a skewed child.
- Fork a non-resident parent → falls back to the snapshot path and matches.

**View disposition (§5.3a's rule) — the one that decides correctness.**
- Mid-turn reprojection **moves** the state: decode ten tokens, force a
  reprojection, decode ten more, and assert the sealed turn's state equals a
  straight-through twenty-token decode. A discard here silently loses the first
  ten tokens' contribution.
- Clean-reprefill **discards**: a thinking turn's sealed state equals what the
  same turn produces with thinking disabled. A move here is the `<think>` skew
  returning.
- The two together are the real test — either alone passes under a
  "always move" or "always discard" implementation, and only the pair pins the
  rule.

**Truncation removal (§5).**
- A grep gate: no call site targets a non-zero block count. The point of the
  deletion is that the API cannot express it, so the test is that the signature
  is gone, not that callers behave.
- `<think>` turn: a conversation of N thinking turns produces the same state as
  the same N turns with thinking disabled at the point of seal — i.e. the
  clean-reprefilled state matches a conversation that only ever saw clean
  tokens. **This is the test that would have caught the site-7 skew**, and it
  needs no reference model: it is the one-shot-vs-segmented shape that settled
  §7.8.
- Sites 1–7: clear-then-rebuild leaves `S` bit-identical across the operation
  (it must be untouched, not reset).

**System prompt (§4.6).**
- Keying: two selector assignments that share an in-tree prefix up to node *n*
  produce the **same** checkpoint at *n* and **different** checkpoints at the
  first node below the differing dim. This is the test that catches a key
  derived from the selection rather than from `pack(selection, ancestor_dims)`.
- Generalisation: the checkpoint computed over the placeholder-substituted
  prefix is identical regardless of which collection members projection
  selects. If it is not, the state and the downstream K/V are approximating the
  collection differently, which is the failure §4.6(2) is written to avoid.
- Glue removal: a projection of a tree with `TreeGlue` nodes emits **zero**
  `ProjectionSegment::Generated` runs after §4.6(1). Assert on the segment
  kinds, not on output text — a text comparison would pass while the tokens
  were being live-prefilled.
- Cold start: a new conversation restoring a branch checkpoint produces the
  same first-turn output as one that actually prefilled the whole prompt.
  This is the §4.6 analogue of the resume oracle and the only test that proves
  the checkpoint is the *right* state rather than merely a consistent one.

**Turn boundary glue (§4.7).**
- `from_flat_grid` with `user_content_start > 0` produces a **real** leading
  `UserStart`, and `validate_tiling` passes. The mechanism is already there; this
  pins it as exercised rather than merely representable.
- Round-trip: seal a turn, reload from the redo log, and assert every `Glue`
  segment's `kv` is `Some` with the same span. This catches a seal anchor that
  did not move (change 4) — the layout would claim a real boundary the chunks
  do not contain.
- **Stream equivalence (§4.7a)** — the load-bearing one. Take a projection of
  N sealed turns, and assert the baked-boundary token stream is **identical**
  to the stream the current `assemble_pieces` emits for the same input. §4.7a
  proves this is exact rather than approximate, so the test is an equality, not
  a tolerance, and it is what pins the both-ends ownership against a later
  "simplification" to one end. The failure it catches — a dropped or doubled
  `<|im_end|>` — reads fine and shifts every boundary after it.
- **No trailing island (§4.7c)** — a projection ending in a live user message
  reserves **zero** gap chunks. Assert on the gap-chunk count, not on output
  text: folding the opener into `NewUserMessage` is invisible in the tokens and
  visible only in whether a gap was reserved.
- **`no_think` split (§4.7b)** — a sealed suppressed turn re-renders its switch
  from its own baked grid, while the *live* turn's switch follows the current
  dial. The regression this guards is a past suppressed turn leaking a stale
  switch onto a later thinking-on turn, which is exactly what the live path's
  comment says must not happen.
- **Baked separators (§4.7d)** — a collection projecting k members emits k
  leading separators and no gap chunks between them; the first member's
  spurious separator is present and accounted for rather than silently
  absorbed.
- Capability dispatch: with boundary K/V present, a glue-capable model still
  produces identical output whether it injects or regenerates. That is the
  proof `Option` is a hit/miss rather than a mode.
- **Zero-glue projection (the gate for the whole section)** — a full projection
  of system prompt + sealed turns + live user message reserves no gap chunks at
  all. This is the single assertion that says the lineage can run; every test
  above is a component of it.

**Provenance fold (§4.8).** The derivations are tier 1 (pure functions over a
synthetic geometry); the two checkpoint-geometry rows are tier 2.
- **Bit-identity on Qwen3-30B** — the derived parameters must produce
  byte-identical folded signatures to today's constants, over a real turn. This
  is the test that permits changing all four at once without re-measuring the
  existing model, and it is the one that makes "backward compatible" checkable
  rather than asserted.
- `rotate_head` at `wph == 2` is byte-identical before and after the word-wise
  generalisation; at `wph == 4` it actually rotates (today it silently returns
  the input, which a naive test would call "passing").
- No all-zero group: folding a 10-layer stack fills all three groups. The
  current constants fail this, so it is a regression test with a known-red
  starting point.
- Fold mismatch is **refused, not scored** — a gallery folded under one
  parameter set and probed under another returns no score and one warning,
  rather than a plausible number.
- `prov_sub_head_dim()` is non-zero at head_dim 256 once `n_palette` is derived,
  so the GPU sign-pack path is taken. Assert the path, not the timing: the
  fallback is correct, so only a path assertion catches the silent demotion.

**End-to-end (the one that would have caught the original bug).** Tier 2 on a
real checkpoint; tier 3 as the zend continuity run.
The oracle that settled §7.8 was *wave one-shot vs wave segmented*, which needs
no reference model. Its analogue here: run a conversation of N turns straight
through; then run the same N turns with a seal + drop + resume in the middle;
assert the same output. A zero state produces fluent, different text — which no
structural check catches, and this does.

Same for fork: parent and fork must produce identical continuations from the
fork point given identical sampling.

Note the tier-1 version of this is **not** the same test. Against
`DummyRecurrentModel` it asserts state equality, which is exact and cheap and
catches every plumbing defect; against a real model it asserts *output*
equality, which is what catches a plumbing-correct state that is nonetheless
the wrong state. Both are worth having, and the cheap one should fail first.

## 7. The open cost question

At 35B geometry — 30 DeltaNet layers, 32 V heads, `d = 128`, `conv_dim = 8192`,
`K = 4`:

```
state / layer     = 32 × 128 × 128 × 4 B  = 2.00 MiB
conv tail / layer = 8192 × 3 × 4 B        = 0.09 MiB
per layer                                 ≈ 2.09 MiB
× 30 recurrent layers                     ≈ 62.8 MiB per snapshot
```

That is the **snapshot** size and it is unchanged by the ping-pong: `export`
reads each slot's `live` buffer only. The *resident* figure is not the same
number — since the ping-pong landed, every slot holds two `s` buffers, so a live
sequence carries ≈123 MiB (2 × 2.00 MiB + 0.09 MiB per layer × 30) rather than
63. Size VRAM from the resident number and the log from the snapshot one.

**~63 MiB per turn seal per conversation**, in a log whose other per-turn
records are kilobytes. Compaction reclaims all but the tail, so steady-state
*storage* is ~63 MiB per live conversation — acceptable. The cost is **write
bandwidth and log growth between compactions**: ten turns is 630 MiB of redo
log, and the export path as written also pays a 63 MiB device→host copy per
seal (`export` does `to_vec1` per layer).

**§4.6 adds a second, differently-shaped multiplier.** Turn snapshots scale
with *live conversations*; branch checkpoints scale with the **selector
cross-product**, which is a build-time constant that does not shrink under
load. A tree with, say, five binary dims and one ternary is 96 leaves ≈ 6 GB at
F32. That changes the weighting below: option 3 (cadence) does nothing for it,
while option 2 (bf16 storage) halves it outright.

Options, none chosen — this is the main thing to decide at review:

1. **Accept it.** Simplest, and correct. Compaction already handles the
   reclaim. Risk is seal latency and log churn on fast turn cadence.
2. **Quantize the stored state.** The payload's per-layer dtype tag exists
   precisely for this. F32 accumulation is mandatory *live* (§7.16 — the state
   is an unbounded running sum); F16/BF16 *storage* is a different claim and
   would halve the cost. It needs the same treatment as the KV C-ladder: a
   measured quality gate, not an assumption. Note this breaks the bit-identical
   resume test, which is a real loss of diagnostic power.
3. **Snapshot every N turns.** Cuts the write rate N×, at the cost of resume
   needing to re-prefill the turns after the checkpoint — which the engine can
   do, but it makes resume latency depend on cadence, and re-prefill is the
   thing the substrate exists to avoid.
4. **Async device→host staging.** Orthogonal to 1–3: pin the export buffer and
   overlap the D2H with the rest of the seal, so the 63 MiB does not sit on the
   seal's critical path. Probably worth doing regardless of which of 1–3 wins.

My recommendation is now **(1) + (4) for turn snapshots, and (2) for §4.6's
branch checkpoints**, measured before it lands. They are separable: the payload
carries a per-layer dtype tag, so the two consumers can store at different
precisions, and the bit-identical resume test survives on the path that matters
most for diagnosing layout bugs.

For turn snapshots the reasoning is unchanged: get it correct and durable
first, with the bit-identical test intact, and treat the size as a follow-up
with its own quality gate rather than a design-time guess. For branch
checkpoints the cost is fixed and paid up front rather than growing with use,
so it is worth measuring bf16 *before* landing rather than after.

## 8. Phasing

Each phase ends green and useful on its own.

| phase | content | gate |
|---|---|---|
| 0 | §4.5 release-on-free + the recycled-slot test | leak gone; recycled slot proven clean |
| 1 | §4.4 `fork_from` device copy | child bit-identical, distinct allocations |
| 2 | **§5.3a views carry the state** — carve forks it, finalize moves it | a multi-turn conversation's `S` advances across turns at all |
| 3 | §5.3 `<think>`: discard instead of rewind; §5.2 rename `_to_blocks`, **delete `_to_tokens`** and rework its four non-zero callers | no path can rewind a slot — asserted over **both** primitives; thinking turns stop skewing; the gate harness's replay control is honest on a hybrid |
| 4 | §4.1 writer + §4.2 distill strip | snapshots on disk; compaction keeps one, drops distilled |
| 5a | §4.7c fold the live opener into `NewUserMessage` | no trailing gap chunk; smallest change, independent of the rest |
| 5b | §4.7 + §4.7a/b bake turn boundaries (both ends; `no_think` split) | stream equivalence vs today, exact |
| 5c | §4.6(1) `TreeGlue` + §4.7d `member_glue` / templates | **zero gap chunks in a full projection** |
| 5d | §4.8(4) derive `n_palette` for head_dim 256 | GPU sign-pack path taken, not the R16 D2H fallback |
| 5e | §4.8(1–3) derive the fold + stamp it on the record | Qwen3-30B bit-identical; hybrid fills all three groups |
| 6 | §4.6(2) per-branch state checkpoints + restore on cold start | a new conversation starts with the prompt in its recurrent state |
| 7 | §4.3 resume + the seal/drop/resume oracle | a restarted conversation continues correctly |
| 8 | §7(4) async staging, if seal latency measures badly | seal latency flat vs Qwen3-30B |
| — | §5.4 spec-decode on forks | out of band; unblocked once phase 1 lands |

**Two phases gate the model switch, for different reasons.** Phase 2 (§5.3a)
is correctness: without it the hybrid's recurrent layers are inert across turn
boundaries, and the model is quietly running on a quarter of its stack. Phase
5c is feasibility: until a projection reserves zero gap chunks, the wave bails
outright on `n_glue > 0`. Everything before phase 2 is prerequisite plumbing;
everything after 5c is durability and quality.

Phase 5 is split because 5a is independently landable and independently
valuable — folding the live opener into `NewUserMessage` removes a gap chunk
that no model needed, on any lineage — while 5b and 5c both require re-sealing
and so want to land together with a rebuild.

Phase 0 remains independent and lands first regardless of what review changes.

## 9. Questions for review

1. ~~`<think>` rewind mechanism~~ — **resolved: (a), fork at turn start**, which
   §5.3a then showed is the existing view machinery. The remaining question is
   narrower: should `finalize_view` *move* the child's state or copy it? A move
   is correct and cheaper, but leaves the child's slot holding a dangling entry
   until release — which §4.5 must then be ordered against.
2. **Restore vs the offset-0 reset** (§4.3) — the survey in §5.1 shows the
   reset never fires on a slot that is cleared-and-rebuilt, because the slot is
   back to `offset > 0` before the next wave. That makes today's behaviour
   correct **by luck**. Does restore get an explicit `seeded` flag, or do we
   make the ordering a stated invariant with a test?
3. **Turn-scoped tombstone** (§4.2) — leave the snapshot and let `turn_index`
   validation reject it, or strip it?
4. **Snapshot size** (§7) — which option, and is the bit-identical resume test
   worth protecting against quantized storage?
5. ~~Join semantics~~ — **resolved in §4.4 / §5.3a**: linear join is a *move* and
   is required on every turn by `finalize_view`; divergent join has no defined
   arithmetic and stays out of scope. See §5.3a's disposition rule for when a
   move happens versus a discard.
6. **Non-hybrid models** — `export_recurrent` returning `None` keeps Qwen3-30B
   free. Confirm no other model in the tree carries recurrent state that should
   ride this.
7. ~~Boundary ownership~~ — **resolved in §4.7a**: both ends, and it is exact
   rather than chosen, because every inter-turn island is literally `AE ++ US`.
8. ~~`member_glue` / template sections~~ — **resolved in §4.7d**: bake against a
   canonical prefix, `member_glue` as a leading separator.
9. **`turn_no_think` stability** (§4.7b) — baking the sealed-turn re-render
   assumes a turn's suppression flag is fixed once sealed. That is what the
   code implies, but it is an assumption the bake would freeze, so it wants a
   test rather than a reading.
10. **The fold shape on a hybrid** (§4.8) — `[n−2,1,1]` is a default carried
    over from a measurement on a uniform 48-layer stack. On the hybrid the top
    two attention layers are 35 and 39, with three recurrent layers between
    them. Needs re-deriving by measurement; the question is whether that
    measurement gates the switch or follows it.
11. **Existing galleries** (§4.8) — stamping the fold makes old records
    self-describing, but records written *before* the stamp exists carry no
    parameters. Treat an unstamped record as Qwen3-30B's fold (the only one
    that ever produced them), or invalidate on the rebuild?

## 10. Decisions taken, so nothing blocks

Every question in §9 has a default here. **Follow the default unless the review
overrides it** — none of them should stop work.

| # | question | default to build against |
|---|---|---|
| 1 | `finalize_view`: move or copy? | **Move.** Cheaper and correct; §4.5's release then clears the child's entry, so order phase 0 first and the dangling entry never exists. |
| 2 | restore vs `offset == 0` reset | **Explicit `seeded` flag** on the store, set by restore/fork, consumed by the first `ensure_recurrent`. Do not rely on the ordering. |
| 3 | turn-scoped tombstone | **Leave the snapshot.** `turn_index` validation on resume rejects it if the recovered turns no longer reach that index. |
| 4 | snapshot size | **Turn snapshots F32; branch checkpoints F32 too — revised, see §11a.13.** Async staging is closed by measurement (§11a.12). bf16 stays P10.5's, behind its quality gate. |
| 5 | join semantics | Resolved: linear = move, divergent = out of scope. |
| 6 | non-hybrid models | `export_recurrent`/`restore_recurrent`/`release_sequence` default to no-ops on the trait. Verify no other model overrides them. |
| 7 | boundary ownership | Resolved: both ends (§4.7a). |
| 8 | `member_glue` / templates | Resolved: bake against a canonical prefix; `member_glue` leading (§4.7d). |
| 9 | `turn_no_think` stability | **Assume fixed once sealed, and add the test** (T3.6 below). If the test fails, fall back to leaving `NoThink` ethereal and treat it as a live-glue exception. |
| 10 | fold shape on a hybrid | **`[n−2,1,1]` as the shipped default.** Measurement follows the switch; it does not gate it. Record the measurement as a follow-up. |
| 11 | unstamped gallery records | **Invalidate on the rebuild.** The substrate is being rebuilt anyway, so do not carry an implicit-fold compatibility path. |

## 11. TODO — the complete, ordered work list

Every change and every test, in dependency order. Each item names the file and
the symbol. **Tier** is the test tier from §6: `1` = fast CPU, `2` = `#[ignore]`
+ GPU/checkpoint, `3` = zend. Nothing here should require a question to be
answered first — §10 supplies a default for every open one.

Paths are repo-relative. Line numbers are from the survey and may drift; the
symbol name is authoritative.

> **The per-item checkboxes below are not a status board — §11a is.**
>
> Phases P-INFRA, P0–P5e, P7 and P9 are implemented and green; the boxes inside
> them were written as a plan and never maintained as one, so most still read
> unticked. The ones ticked below are the items verified against a running model
> in this campaign, and they are ticked because they were *run*, not because a
> header says so.
>
> Ticking the rest would be worse than leaving them, because several items are
> **superseded rather than done** — the design said something and the code said
> otherwise, and a tick would assert the design's version:
>
> | Item | What actually happened |
> |------|------------------------|
> | P3.4 | "Remove the block-count parameter (it only ever takes 0)" is false — two callers pass non-zero legitimately. The parameter stays (§11a.1). |
> | P3.1 | The rewind is unreachable via a capability gate, not deleted (§11a.1). |
> | P5a | Not independently landable; landed with P5b, and the bake is central (§11a.2). |
> | P5c | Two thirds of it has no production caller and was left alone (§11a.2b). |
> | P5e.1a | Named both gather paths; **both were missed** until this campaign (§11a.6). |
> | P6.1 | Its premise — that the state falls out of a prefill the generator already runs — is false (§11a.3b). |
> | T5b.5 | Not implementable as written (§11a.10). |
>
> Read §11a first, then this list as the map of what was *intended*.

---

### P-INFRA — test harness (do first; everything else asserts through it)

- [ ] **I1.** Add `DummyRecurrentModel` to `candle-conversation/src/scheduler/mod.rs`
      tests, beside `DummyModel` (~:7935). Wraps `DummyModel`, adds a toy state:
      2 layers × 2×2 F32 per sequence, in a `HashMap<usize, [[f32;4];2]>`.
      Implements the three hooks added in P0/P2/P4/P7 as they land.
- [ ] **I2.** Add `make_test_scheduler_recurrent()` mirroring
      `make_test_scheduler()` (~:8057) but with `DummyRecurrentModel`.
- [ ] **I3.** Helper `assert_state_eq(scheduler, seq, expected: [[f32;4];2])` —
      exact equality, no tolerance. Every tier-1 state assertion goes through it.

---

### P0 — release on free  *(independent; fixes a live leak + correctness bug)*

**Changes**
- [ ] **P0.1.** `candle-transformers/src/models/batched_inference.rs`: add to
      `ManagedBatchedModel`
      `fn release_sequence(&self, _seq: usize) -> Result<()> { Ok(()) }`
      (default no-op), beside `truncate_sequence` (~:3252).
- [ ] **P0.2.** `candle-transformers/src/models/qwen35/forward.rs`: override it
      on `HybridBatched` → `self.release_recurrent(seq)`.
- [ ] **P0.3.** `candle-conversation/src/scheduler/mod.rs`: call
      `model.release_sequence(idx)` at **both** `session.free_sequence` sites —
      `:3028` (`FreeSequence` handler) and `:4970` — and at `:7544`
      (`free_sequence(view_id.0)` on reprojection).

**Tests** (tier 1)
- [ ] **T0.1.** Free a slot, allocate a new sequence that lands on the same slot
      index, first wave carries `offset > 0` → the new sequence's state is
      **zeros**, not the previous conversation's. This is the recycled-slot
      defect; it should fail before P0.3 and pass after.
- [ ] **T0.2.** `recurrent_len()` returns to its pre-allocation value after free.

---

### P1 — `fork_from`  *(the primitive P2/P7 need)*

**Changes**
- [ ] **P1.1.** `candle-transformers/src/models/delta_net/state_store.rs`
      (`RecurrentStateStore`, `:129`): add
      `pub fn fork_from(&self) -> Result<Self>` — new slots, same `dims`/`hash`/
      `layer_index` order, `open: false`. Per slot: `live` is a
      `DeltaNetState::snapshot()` of the parent's `live` (device-to-device);
      `backup` is a fresh allocation the kernels fully overwrite, so it needs no
      copy; `advanced: false`.
- [ ] **P1.2.** Refuse `fork_from` while `self.open` (mid-wave) — same rule as
      `export`. Sharper than it was before the ping-pong: mid-wave, the parent's
      *advanced* state lives in `backup` and its `live` is one wave stale, so a
      mid-wave fork would silently copy the wrong buffer rather than merely
      copying a moving one.
- [ ] **P1.3.** Build the fork through `layer_state` / the slot fields directly,
      **never** `layer_state_pair_mut` — that accessor sets `advanced`, and a
      fork that trips it makes the parent's next `commit_wave` swap in a buffer
      no wave wrote.

**Tests** (tier 1 for shape, tier 2 for device)
- [ ] **T1.1.** Child's `s` and `conv_tail` are bit-identical to the parent's.
- [ ] **T1.2.** Buffers are **distinct allocations**: mutate the child, re-read
      the parent, assert unchanged. This is the test that catches a `Clone`
      sharing storage — the hazard `DeltaNetState`'s missing `Clone` prevents.
      Assert it on **both** halves of the ping-pong: `layer_state_pair` hands out
      a `Tensor` clone of the write buffer, so a fork that shares that one looks
      correct until the child's first commit swaps it into the parent's view.
- [ ] **T1.3.** `fork_from` mid-wave errors.
- [ ] **T1.4.** Forking does not disturb the parent's wave bookkeeping: fork a
      parent whose slots are all `advanced: false`, then commit a wave on the
      parent and assert the state it installs is the one its own wave wrote
      (P1.3's hazard).
- [ ] **T1.5.** A child forked from a parent mid-*turn but between waves* carries
      the parent's committed state, not its write buffer — i.e. the fork reads
      `live` after the swap, never `backup`.
- [x] **T1.6.** *(tier 2)* On CUDA, the copy never touches host memory —
      assert via device pointers differing and no H2D/D2H in the span.

---

### P2 — views carry the state  ***(GATES THE SWITCH)***

**Changes**
- [ ] **P2.1.** `batched_inference.rs`: add to `ManagedBatchedModel`
      `fn fork_recurrent(&self, _parent: usize, _child: usize) -> Result<()> { Ok(()) }`
      and
      `fn move_recurrent(&self, _child: usize, _parent: usize) -> Result<()> { Ok(()) }`.
- [ ] **P2.2.** `qwen35/batched.rs`: implement both on the recurrent map —
      `fork_recurrent` inserts `parent.fork_from()` under `child`;
      `move_recurrent` removes `child`'s store and inserts it under `parent`.
- [ ] **P2.3.** `candle-transformers/src/models/delta_net/state_store.rs`
      (`RecurrentStateStore`, `:129` — **not** `batched.rs`, which holds only the
      map): add `seeded: bool` with `mark_seeded()` / `take_seeded()`. It is a
      store-level flag like `open`, not a per-slot one like `advanced`. Set by
      `fork_recurrent` and by restore; `ensure_recurrent`
      (`qwen35/batched.rs:175`) **skips the `offset == 0` reset exactly once**
      when it is set, then clears it. (§10 decision 2.)
- [ ] **P2.4.** `scheduler/mod.rs::create_view` (~:6699): after
      `create_view_sequence` returns the view id, call
      `model.fork_recurrent(parent_id.0, view_id.0)`.
- [ ] **P2.5.** `scheduler/mod.rs` `:5358` (`finalize_view` at seal): call
      `model.move_recurrent(view_id.0, parent_id.0)` **before**
      `free_sequence(view)`.
- [ ] **P2.6.** `scheduler/mod.rs` `:7544` (reprojection): this path finalizes —
      call `move_recurrent` before the free. **Do not** discard here (§5.3a
      disposition rule).

**Tests** (tier 1)
- [ ] **T2.1.** A three-turn conversation's state at turn 3 differs from a
      one-turn conversation's — i.e. state advances across turn boundaries at
      all. Fails on today's code.
- [ ] **T2.2.** **Disposition pair (both required).**
      (a) Decode 10 tokens, force a reprojection, decode 10 more → sealed state
      equals a straight-through 20-token decode. *(move)*
      (b) A `<think>` turn's sealed state equals the same turn with thinking
      disabled. *(discard — lands with P3)*
      Either alone passes under a naive always-move/always-discard build.
- [ ] **T2.3.** `move_recurrent` leaves no entry under the child id.
- [ ] **T2.4.** The `seeded` flag suppresses exactly one reset: a forked slot
      whose first wave carries `offset == 0` keeps its forked state; its
      *second* wave at `offset == 0` resets normally.

---

### P3 — delete rewind, discard instead

**Changes**
- [ ] **P3.1.** `batched_inference.rs`: rework the spec-decode accept path
      (~:3482) per §5.4 — fork before the verify block, discard on partial
      accept, advance a fresh fork over the `k` accepted tokens. If spec-decode
      is out of scope for the switch, instead **assert** it is disabled for
      recurrent models and remove the call.
- [ ] **P3.2.** `batched_inference.rs`: **delete** `ManagedBatchedModel::truncate_sequence`
      (`:3252-3259`) and both overrides — `qwen35/forward.rs:190` and
      `latent_moe/wave.rs:910` (deepseek4's, which also drops
      `rollback_verify_state`'s caller; fold that into P3.1's fork rework).
- [ ] **P3.3.** `qwen35/forward.rs`: delete the `tokens != 0` bail (~:182) — the
      API can no longer express it.
- [ ] **P3.4.** `candle-nn/src/kv_cache/chunked/sequence_ops.rs`: rename
      `truncate_sequence_to_blocks` → `clear_slot_chunks` and **remove the
      block-count parameter**. This is only possible once P3.4b rewrites site 9,
      which passes a non-zero count today — the parameter is *not* already
      0-only.
- [ ] **P3.4a.** **Delete `truncate_sequence_to_tokens` outright** — the backing
      method (`sequence_ops.rs`) and the session wrapper
      (`batched_inference.rs:1275`). This is the primitive that actually
      expresses a rewind; P3.4's block-count rename does not touch it, and
      leaving it means phase 3's gate is false. Its callers are P3.4b–P3.4e.
- [ ] **P3.4b.** `batched_inference.rs:1211` (site 9, `reserve_glue_gap`
      rollback): the non-zero `_to_blocks` target. Replace with an explicit
      release of the chunks this call reserved, so the rollback stops being
      spelled as a truncation.
- [ ] **P3.4c.** `candle-nn/src/kv_cache/chunked/alloc.rs:1722` (site 11,
      `heal_tail_divergence`): the heal trims a failed wave's surplus token.
      Decide against §5.3a's disposition rule — the wave rolled back, so `S` is
      already at the entry value and the trim is consistent — and **state that
      in the code**, gated by T3.7 rather than left as a reading.
- [ ] **P3.4d.** `candle-nn/src/kv_cache/cache.rs:711`
      (`truncate_chunked_to_tokens`, site 12): remove with its only caller.
- [ ] **P3.4e.** `batch_test/utils.rs` sites 13 (`:580`,
      `decode_replay_probe`) and 14 (`:1182`, the repeat loop). **These are live
      defects on the gate harness, not just renames** — both claim to restore
      identical state and neither touches the recurrent store, so on a hybrid
      the control drifts and reports its own drift as model non-determinism.
      Replace each with a fresh sequence per replay/repeat (`create_sequence` +
      re-prefill), which restores identical state on every model rather than
      only on models with no state to restore.
- [ ] **P3.5.** `scheduler/mod.rs` `:5490` (site 8, `<think>` clean-reprefill):
      replace the partial truncate with **discard the view** — do not
      `move_recurrent`, free the view, re-prefill clean onto the untouched
      parent.
- [ ] **P3.6.** Update the seven zero-target call sites to the renamed API:
      `mod.rs:4646`, `:5622`, `:5731`, `:7551`;
      `projection_assembler.rs:625`; `prefill.rs:2475`;
      `batched_inference.rs:1399` (the seal quantize path — session-level, and
      the one the scheduler-scoped survey missed). Add a one-line comment
      at each stating the slot is repopulated before the next wave, which is
      *why* the recurrent state is untouched.

**Tests**
- [ ] **T3.1.** *(tier 1)* Grep gate, over **both** families: neither
      `truncate_sequence_to_blocks`'s block-count parameter nor
      `truncate_sequence_to_tokens` exists anywhere in the tree. A gate phrased
      only as "no call site passes a non-zero block count" passes green with the
      token form and all four of its non-zero callers intact, which is the
      failure this wording exists to prevent.
- [ ] **T3.2.** *(tier 1)* T2.2(b) now passes — the `<think>` discard.
- [ ] **T3.3.** *(tier 1)* Clear-then-rebuild leaves `S` **bit-identical**
      across the operation (untouched, not reset) at all seven zero-target sites.
- [x] **T3.4.** *(tier 2)* `<think>` oracle on a real checkpoint: N thinking
      turns produce the same sealed state as N with thinking disabled.
- [ ] **T3.5.** *(tier 1)* Error path (site 6): a failed reprefill discards the
      view and leaves the parent's state at the turn boundary.
- [ ] **T3.6.** *(tier 1)* **`turn_no_think` stability** (§10 decision 9): a
      sealed turn's suppression flag is unchanged by later projections. If this
      fails, `NoThink` stays ethereal — see P5b.3.
- [ ] **T3.7.** *(tier 1)* **Heal-after-failed-wave consistency** (P3.4c): fail a
      wave mid-flight, let the rollback + tail heal run, and assert the slot's
      `S` matches the wave-entry value **and** its KV length matches the
      delivered offset. Pins the one path that trims a slot rather than clearing
      it, which §5.3a's disposition table does not otherwise cover.
- [x] **T3.8.** *(tier 2)* **The harness control is honest again** (P3.4e): run
      `decode_replay_probe` against the hybrid and assert zero divergences. On
      today's code this is red for the hybrid and green for Qwen3-MoE — which is
      exactly the shape that would otherwise be read as "the hybrid is
      non-deterministic."

---

### P4 — snapshot writer + distill strip

**Changes**
- [ ] **P4.1.** `batched_inference.rs`: add
      `fn export_recurrent(&self, _seq: usize) -> Result<Option<SnapshotPayload>> { Ok(None) }`.
      (Payload type moves to a shared crate or is mirrored — see P4.2.)
- [ ] **P4.2.** Decide the type boundary: `candle-conversation` depends on
      `candle-transformers`, not the reverse, so the hook returns
      `Option<Vec<ExportedLayerState>>` + `schedule_hash` and the scheduler
      assembles `SnapshotPayload`. Do **not** move `SnapshotPayload` down.
- [ ] **P4.3.** `qwen35/forward.rs`: implement via `RecurrentStateStore::export()`;
      assert `!store.open` (the seal runs outside the wave — make it an
      assertion, not an assumption).
- [ ] **P4.4.** `scheduler/mod.rs` (~:6390, beside the `WideQSig` write, **before**
      the `Tokens` enqueue): build the payload, stamp `turn_index` from the
      sealing turn, call `conversation.enqueue_recurrent_snapshot(timeline, payload.encode())`.
- [ ] **P4.5.** `candle-conversation/src/persistence/compaction.rs` (~:154): gate
      the `recurrent_snapshot_entries()` loop on the `distilled` map built ~25
      lines below, so a distilled timeline's snapshot is dropped.
- [ ] **P4.6.** Verify the timeline-tombstone removal still fires for a
      tombstoned-**and**-distilled timeline (which deliberately escapes the
      wholesale drop).

**Tests** (tier 1 — the persistence harness is already CPU-only)
- [ ] **T4.1.** Two snapshots for one timeline: the second supersedes,
      compaction keeps one, and the surviving bytes are the **second**.
- [ ] **T4.2.** A distilled timeline's snapshot is dropped; its `WideQSig`
      survives. (Inverse of the existing distillation test.)
- [ ] **T4.3.** A tombstoned timeline's snapshot is dropped.
- [ ] **T4.4.** Plan/execute race: a snapshot appended between plan and execute
      is the survivor; the planned copy is not relocated. (Extend the existing
      `superseded_snapshot_is_not_relocated`.)
- [ ] **T4.5.** `SnapshotPayload` round-trip at real 35B geometry — 30 layers ×
      32 heads × 128², extending the existing tiny-dims test.
- [ ] **T4.6.** Export mid-wave errors.

---

### P5a — fold the live opener into `NewUserMessage`

**Changes**
- [ ] **P5a.1.** `scheduler/mod.rs` (~:2802): delete the
      `Generated { user_start_current }` segment push.
- [ ] **P5a.2.** `scheduler/mod.rs` (~:2840): delete the live `no_think`
      `Generated` push.
- [ ] **P5a.3.** Prepend `user_start` (+ `no_think` when the dial suppresses) to
      the `ProjectionSegment::NewUserMessage { tokens }` payload instead. The
      live `no_think` stays dynamic because these tokens are rebuilt every
      projection.

**Tests** (tier 1)
- [ ] **T5a.1.** A projection ending in a live user message reserves **zero**
      gap chunks. Assert on gap-chunk count, not tokens — the fold is invisible
      in the token stream.
- [ ] **T5a.2.** The emitted token stream is unchanged vs today.
- [ ] **T5a.3.** Toggling the no-think dial changes the live turn's tokens on
      the *next* projection (still dynamic).

---

### P5b — bake turn boundaries

**Changes**
- [ ] **P5b.1.** Reserve grid room for the leading marker: the turn's own
      prefill must cover its `user_start` tokens so `from_flat_grid` receives
      `user_content_start > 0`.
- [ ] **P5b.2.** Same at the tail for the closing `ImEnd` (= `assistant_end`).
- [ ] **P5b.3.** `NoThink`: bake the **sealed-turn re-render** (fixed per turn);
      the **live** switch stays dynamic via P5a.3. If T3.6 failed, skip the bake
      and leave `NoThink` ethereal.
- [ ] **P5b.4.** Move the seal anchor: `turn_start_parent_blocks` must anchor
      **before** the leading marker, or the baked boundary is outside the sealed
      range.
- [ ] **P5b.5.** `projection_assembler.rs::assemble_pieces` (~:290): stop
      wrapping `Sealed::Turn` — no `run += user_start`, no `run += assistant_end`.
      A sealed turn emits alone.
- [ ] **P5b.6.** `batched_inference.rs`: add `can_gap_fill: bool` to
      `ModelCoreProperties`; `false` for the hybrid. The assembler reads it
      before planning rather than discovering it via `forward_wave`'s bail.
- [ ] **P5b.7.** **The compression turn must bake its own head/tail.**
      `scheduler/mod.rs` ~:4192 builds a summary/compression turn as
      `[user][user_end][assistant_start][assistant]` with the comment *"No
      leading `no_think` / `user_start` head: those are live `Generated`
      segments the assembler re-emits around the sealed turn."* P5b.5 stops the
      assembler re-emitting, so without this the compression turn silently
      loses its opener. Give it the same baked head + tail as a normal turn.
- [ ] **P5b.8.** **Check the `TurnHalf` injection for double framing.**
      `projection_assembler.rs` ~:311 injects a turn-half with no wrapping
      because *"the compression pass supplies its own framing glue."* With the
      turn's markers baked, verify the injection windows on `user_span()`
      (which excludes the leading marker by construction, since
      `user_content_start` is the `User` segment's offset) and not on the whole
      turn grid. If it windows the whole grid, the marker and the compression
      pass's own framing both land.
- [ ] **P5b.9.** Update the persisted-turn contract doc at `substrate.rs`
      ~:738-756, which currently states the head and tail are *"**not**
      persisted."* That comment is the contract; leaving it stale makes the next
      reader trust the wrong thing.

**Tests** (tier 1)
- [ ] **T5b.1.** **Stream equivalence** — the baked-boundary token stream is
      *identical* to what today's `assemble_pieces` emits for the same input.
      Equality, not tolerance (§4.7a proves it is exact). This is the test that
      pins both-ends ownership against a later "simplify to one end."
- [ ] **T5b.2.** Concatenation: two consecutive sealed turns yield exactly one
      `<|im_end|>` at the join.
- [ ] **T5b.3.** Round-trip: seal, reload from the redo log, assert every `Glue`
      segment's `kv` is `Some` with the same span. Catches a seal anchor that
      did not move (P5b.4).
- [ ] **T5b.4.** `from_flat_grid` with `user_content_start > 0` passes
      `validate_tiling`. (Extends the existing `leading_boundary_offsets_user_body`.)
- [ ] **T5b.5.** Capability dispatch: with boundary K/V present, a
      `can_gap_fill: true` model produces identical output whether it injects or
      regenerates — proving `Option` is a hit/miss, not a mode.
- [ ] **T5b.6.** A compression / summary turn round-trips with a baked head and
      tail (P5b.7), and its projected stream has exactly one opener — not zero
      (assembler stopped, builder not updated) and not two (both fired).
- [ ] **T5b.7.** A `TurnHalf` injection contains the user body **without** the
      leading marker (P5b.8), so the compression pass's own framing is the only
      framing present.

---

### P5c — `TreeGlue`, `member_glue`, templates

**Changes**
- [ ] **P5c.1.** `candle-conversation/src/projection/schema.rs`: give `TreeGlue`
      nodes sealed variants per branch they are active in
      (`TreeGlue::active_keys` already records which). They stop being
      prefix-transparent, so everything below them re-seals.
- [ ] **P5c.2.** `SectionCollection::member_glue`: bake as a **leading**
      separator on every member's seal. Accept the spurious copy on the first
      selected member.
- [ ] **P5c.3.** `is_template` sections (`project.rs` ~:1988): seal against a
      canonical prefix rather than emitting `Generated`.
- [ ] **P5c.4.** Mark `SlotState.glue_islands` dead for `can_gap_fill: false`
      models — do not leave an apparently-live cache that never fills.

**Tests** (tier 1)
- [ ] **T5c.1.** **Zero gap chunks in a full projection** — system prompt +
      sealed turns + live user message. This is the single assertion that says
      the lineage can run; every other glue test is a component of it.
- [ ] **T5c.2.** A collection projecting *k* members emits *k* leading
      separators and no gap chunks between them; the first member's spurious
      separator is present and accounted for.
- [ ] **T5c.3.** A tree projection emits zero `ProjectionSegment::Generated`.
      Assert on segment kinds, not text.

---

### P5d — `n_palette` for head_dim 256

**Changes**
- [ ] **P5d.1.** `batched_inference.rs::prov_sub_head_dim` (~:1942): replace the
      `head_dim / N_PALETTE, must be ≤ 32` rule with a derived
      `n_palette = smallest p such that head_dim / p ≤ 32`.
- [ ] **P5d.1a.** `batched_inference.rs:1934`: the same function stamps
      `n_palette: candle_nn::kv_cache::N_PALETTE` into `ProvSignPacked`, three
      lines below the `sub_head_dim` P5d.1 just derived. Derive it from the same
      rule. **Landing P5d.1 without this is worse than landing neither** — the
      struct then carries two contradictory numbers and a consumer computing
      `n_palette × sub_head_dim` reads 128 for a 256-dim head.
- [ ] **P5d.2.** Thread a per-backing `n_palette()` through the arena/table path
      (the 16-band latent precedent in `arena_table.rs` and `backing.rs` shows
      the shape). 128 → 4, 256 → 8.
- [ ] **P5d.3.** Verify `assemble_folded_prov_sigs` (`scheduler/mod.rs` ~:630)
      reads `n_palette` from the packed struct rather than the constant.

**Tests**
- [ ] **T5d.1.** *(tier 1)* `prov_sub_head_dim()` is 32 at head_dim 128 (4
      bands) and 32 at head_dim 256 (8 bands) — never 0.
- [x] **T5d.2.** *(tier 2)* At head_dim 256 the **GPU sign-pack path is taken**.
      Assert the path, not the timing: the fallback is correct, so only a path
      assertion catches the silent demotion.
- [ ] **T5d.3.** *(tier 1)* Band packing is bit-identical at head_dim 128
      before/after.

---

### P5e — derive the provenance fold

**Changes**
- [ ] **P5e.1.** `provenance/wide_sig.rs`: replace `PROV_HEADS_PER_LAYER` with a
      parameter carried on the fold call — the model's `n_kv_head`.
- [ ] **P5e.1a.** Thread it through **both** gather paths — they are separate
      code and either can be missed:
      the GPU fast path `assemble_folded_prov_sigs`
      (`scheduler/mod.rs` ~:686, `fold_provenance(&WideQSig { n_heads, words })`)
      **and** the CPU fallback in `gather_wide_sigs`
      (~:6682, `fold_provenance(&WideQSig::from_band(&band, head_dim))`).
      The two are asserted bit-identical today; keep that true.
- [ ] **P5e.2.** Replace `PROV_FOLD_SIZES` with a derivation `[n − 2, 1, 1]` over
      the capture-layer count. Refuse to emit an all-zero group (return an
      error rather than a signature that cannot be filled).
- [ ] **P5e.3.** Replace `PROV_FOLD_SHIFT` with `head_dim / 4`.
- [ ] **P5e.4.** Generalise `rotate_head` to a word-wise rotate over `wph` words
      (currently bails unless `wph == 2`). Its return type must change too — the
      current `(u64, u64)` cannot carry a `wph == 4` result at all.
- [ ] **P5e.4a.** **Delete `fold_provenance`'s own `if wph == 2` branch**
      (`wide_sig.rs:124-131`), which XORs the *unrotated* `raw.words` on the
      `else` arm. Without this P5e.4 is invisible: the generalised rotate runs,
      its output is dropped, and the fold still has no stagger at head_dim 256.
      A test written against `rotate_head` (T5e.1) passes either way, so the
      assertion that catches it is T5e.6, on the fold.
- [ ] **P5e.5.** `provenance/scan.rs`, `gpu.rs`, `packed.rs`: replace the
      `HEADS_PER_GROUP` alias with the value **the signature was folded with**,
      read from the record.
- [ ] **P5e.5a.** `zend/examples/provenance_layers.rs:34` holds a **fourth**
      copy — a bare `const HEADS_PER_GROUP: usize = 4`, not an alias, so it does
      not turn up in a `PROV_HEADS_PER_LAYER` grep. Derive it from the probed
      signature's own shape. Left alone, `project_groups` strides past the end
      of a hybrid signature, its `e <= words.len()` guard drops the group, and
      the harness scores an empty projection as a valid one — in a tool whose
      only output is which layers carry the signal.
- [ ] **P5e.6.** Stamp the fold parameters (`n_kv_head`, group sizes, shift,
      `head_dim`) into the `WideQSig` record; bump its payload version.
- [ ] **P5e.7.** **Refuse to score across a fold mismatch**, with a
      distinguishable WARN. Never score, never silently coerce.
- [ ] **P5e.8.** Treat unstamped records as invalid (§10 decision 11) — the
      substrate rebuild regenerates them.

**Tests**
- [ ] **T5e.1.** *(tier 1)* `rotate_head` is byte-identical at `wph == 2` before
      and after; at `wph == 4` it **actually rotates** (today it silently
      returns the input, which a naive test would call passing).
- [ ] **T5e.2.** *(tier 1)* Folding a 10-layer stack fills all three groups.
      Known-red on today's constants.
- [ ] **T5e.3.** *(tier 1)* Fold mismatch is refused, not scored: one warning,
      no score.
- [x] **T5e.4.** *(tier 2)* **Bit-identity on Qwen3-30B** — derived parameters
      produce byte-identical folded signatures to today's constants over a real
      turn. This is what permits changing all five at once.
- [x] **T5e.5.** *(tier 2)* Real-geometry fold on the 35B: 10 attention layers,
      2 KV heads, head_dim 256 → 3 groups × 2 heads × 256 bits = 1536 bits.
- [ ] **T5e.6.** *(tier 1)* **The fold staggers at `wph == 4`** (P5e.4a): fold a
      synthetic raw signature whose group-0 layers are *identical*, at head_dim
      256, and assert the group head is **not** zero. Identical layers cancel
      under XOR without a stagger, so this is red on today's code and stays red
      if only `rotate_head` is generalised — which T5e.1 cannot distinguish,
      because it never runs the fold.

---

### P6 — system-prompt branch checkpoints

**Changes**
> **P6 landed, but not as written below — read §11a.11 first.** The checkpoint
> is keyed by content prefix rather than `pack` (P6.3's key goes stale on any
> prompt edit), a conversation computes only the branch it runs on rather than
> the cross-product (the live tree is 200 branches, not 96), and P6.2 is retired
> because there is no cross-product left to accelerate.

- [x] **P6.1.** Generator — the section-seal path,
      `conversation.rs::insert_section_collection` (~:731) and its
      `_with_progress` variant, which is what pre-seals the tree's variants.
      For each `SectionTree` leaf (full selector assignment), compute the
      recurrent state over the branch's ordered in-tree prefix, with collection
      nodes substituted by their `inject_collection` placeholder (§4.6(2)).
      The state is a by-product of a prefill the generator already runs — do
      not add a second pass.
- [~] **P6.2.** *(retired — §11a.11)* Use per-node-per-branch checkpoints as a **build-time**
      accelerator only — resume the walk from the deepest shared ancestor.
      Runtime looks up leaves only.
- [x] **P6.3.** *(keyed by content prefix instead — §11a.11)* Store keyed by `SectionTree::pack(selection, dims.len())`, in
      bf16 (§10 decision 4).
- [x] **P6.4.** *(after priming, not in `create_sequence` — §11a.11a)* On cold start, restore the branch checkpoint
      for the active assignment before the first wave; set `seeded`.

**Tests**
- [x] **T6.1.** *(tier 1)* Two assignments sharing an in-tree prefix up to node
      *n* share the checkpoint at *n* and differ at the first node below the
      differing dim. Catches a key derived from the selection instead of
      `pack(selection, ancestor_dims)`.
- [x] **T6.2.** *(tier 1)* The checkpoint is invariant to which collection
      members projection selects.
- [x] **T6.3.** *(tier 3; caught two real defects on its first run — §11a.11)* Cold start restoring a branch checkpoint produces the
      same first-turn output as one that actually prefilled the whole prompt.

---

### P7 — resume

**Changes**
- [ ] **P7.1.** `batched_inference.rs`: add
      `fn restore_recurrent(&self, _seq: usize, _layers: &[ExportedLayerState], _hash: u64) -> Result<bool> { Ok(false) }`.
- [ ] **P7.2.** `qwen35/forward.rs`: implement via `RecurrentStateStore::import`;
      set `seeded` on success.
- [ ] **P7.3.** `scheduler/mod.rs::create_sequence` (~:2476–2526): for a slot
      binding to a timeline with a snapshot, read the payload via
      `recurrent_snapshot_loc` + the persistence handle, decode, restore.
- [ ] **P7.4.** `conversation.rs::fork_onto` (~:2501): add
      `parent: Option<SequenceId>` to `SchedulerRequest::NewSequence`; prefer
      `fork_recurrent` from a live parent **only when the fork's target timeline
      is the parent's own** — that is the one case where the parent's live state
      describes the history the child will hold. Every other fork keeps the
      snapshot read. *(As first written this item said "prefer the live parent"
      unconditionally, and the implementation faithfully did: the daemon resumes
      a client by forking its base conversation onto the client's timeline, so
      every resumed conversation ran on the base conversation's memory, copied
      over its own correctly-restored snapshot. A1 of
      `recurrent_state_behavioural_tests.md` caught it; nothing else could — the
      K/V was right and the conversation read perfectly. The predicate is
      derived in `fork_onto`, not passed by callers, and unit-asserted in
      `fork_inherits_history_tests`. A1 then found the second half of the same
      lesson: a resume restores THREE pieces of durable state — K/V, recurrent
      memory, and the carried selection belief. `Scheduler::restore_carried_belief`
      rebuilds the third from the recovered turns' persisted projection events;
      see the A1 note in `recurrent_state_behavioural_tests.md`.)*
- [ ] **P7.5.** Refuse `fork()` while `turn_in_flight` (§4.4 boundary
      constraint) with an explicit error.
- [ ] **P7.6.** Every rejection path logs at **WARN with a distinguishable
      reason**: hash mismatch / `turn_index` too new / no snapshot.

**Tests**
- [ ] **T7.1.** *(tier 1)* Seal → drop store → resume → state bit-identical.
- [ ] **T7.2.** *(tier 1)* Hash mismatch → zeros **and** a WARN.
- [ ] **T7.3.** *(tier 1)* `turn_index` beyond the recovered turns → zeros + WARN.
- [ ] **T7.4.** *(tier 1)* Fork a non-resident parent → snapshot path, matching
      state.
- [ ] **T7.5.** *(tier 1)* Fork with a turn in flight → explicit error.
- [x] **T7.6.** *(tier 2)* Parent and fork produce identical continuations from
      the fork point under identical sampling.

---

### P8 — async staging *(only if seal latency measures badly)*

- [~] **P8.1.** *(not needed — the export is 4.06 % of turn wall, §11a.12)* Pin
      the export buffer; overlap the D2H with the rest of the seal.
- [x] **T8.1.** *(measured: 39.8 ms/seal, 4.06 % of wall — §11a.12)* Seal latency
      on the hybrid is flat against Qwen3-30B.

---

### P9 — zend  *(tier 3, the last gate)*

- [x] **P9.1.** `zend/tests/`: hybrid **ingest** test — a workspace ingests and
      seals; snapshots written; compaction keeps one tail per conversation;
      distilled timelines shed theirs.
- [x] **P9.2.** Hybrid **continuity** test, `infinite_conversation_smoke` shape:
      turns accumulate, recall works, and **the recurrent state is non-zero at
      depth** (assert the state, not just the text — silent zeroing reads as
      fluent amnesia).
- [x] **P9.3.** **Restart** test: stop the daemon, restart, `fork_resuming` the
      timeline, continue coherently. The only test exercising resume against a
      real redo log.
- [x] **P9.4.** All three `#[ignore = "…"]` with cost + run instructions,
      matching the existing convention.

---

### P10 — follow-ups (not gating)

- [ ] **P10.1.** Measure the fold shape on the hybrid (§10 decision 10) and
      re-derive `[n−2,1,1]` if the top-two-attention-layer assumption does not
      hold at 35/39.
- [ ] **P10.2.** Measure whether attention-only capture retains retrieval
      quality (risk 5 / Phase 4 exit criterion).
- [ ] **P10.3.** Spec-decode on forks (§5.4), if P3.1 deferred it.
- [ ] **P10.4.** Zero-glue vs fork-glue ablation — also a direct read on how
      much the recurrent memory contributes. Extend it to cover the two baked
      approximations this design accepts: boundary markers sealed under their
      seal-time prefix rather than the runtime one (§4.7's "reason they were
      never baked"), and `member_glue` as a fixed leading separator (§4.7d).
      Both are defensible by argument; neither has been measured.
- [ ] **P10.5.** bf16 quality gate for turn snapshots (§7 option 2) if the F32
      write rate measures badly.
- [ ] **P10.6.** **Measure per-turn fork traffic** (§4.4). The ping-pong store
      removed the per-wave copy this cost used to be amortised against, and P2
      puts a ~63 MiB device copy on every dialogue turn. If turn latency
      regresses after P2, this is the first place to look — and the question it
      raises is whether a view's fork can be deferred to the first layer that
      actually advances, since a turn that reprojects before decoding anything
      has paid for a copy nothing read.

---

### Definition of done — **met**

All four conditions hold. The switch is ready when:

1. **T2.1 and T2.2 pass** — state advances across turns, and disposition is
   correct in both directions. *(Correctness gate.)*
2. **T5c.1 passes** — a full projection reserves zero gap chunks.
   *(Feasibility gate.)*
3. **T5e.4 passes** — Qwen3-30B folds bit-identically. *(No regression on the
   outgoing model.)*
4. **P9.1–P9.3 pass** — zend ingests, converses, and survives a restart.

Everything else is quality or durability and may land after the switch.

## 11a. What implementation found — corrections to this document

Four things below are **corrections**, not progress notes: the design said
something and the code said otherwise. Each is recorded with the evidence,
because the reasoning that produced the wrong answer is still in the sections
above and would produce it again.

### 11a.1 The rewind cannot be deleted; it can be made unreachable (P3)

§5.2 says `truncate_to_tokens` should be "deleted outright" because "every
caller is a rewind". Two of them are not, and a third cannot be rewritten
tonight or safely:

- **`heal_tail_divergence`** (`alloc.rs:1722`) trims a failed wave's surplus
  token so the layers agree with the offset the session already delivered. That
  is a repair toward the committed state, not a rewind away from it.
- **`KvCache::truncate_to_offset`** (`cache.rs:1280`) is public API on the simple
  contiguous/chunked cache, unrelated to the batched paged path.
- **deepseek4's speculative verify** carries a compressor + gallery rollback
  attached to the same hook (`latent_moe/wave.rs:910`). Rewriting it to the
  fork-based shape §5.4 describes is real work in a GPU-only, performance-
  critical path with no CPU-runnable gate.

**What landed instead.** The operation is now *unreachable* for a model that
cannot express it, which is the goal §5.2 actually states ("stops being guarded
and starts being inexpressible") reached by a different route:

- `ManagedBatchedModel::carries_recurrent_state()` — a real model property, like
  `head_dim`. `true` for `HybridBatched`.
- `speculative_decode_step_batch` refuses at its **entry point** when that holds,
  naming §5.4 and P10.3. Not inside the rewind: by the time the driver has
  drafted and verified, refusing is an error against work already done.
- `truncate_sequence` → renamed `rewind_after_verify`, so the hook names its one
  purpose instead of reading like a general lifecycle method.
- The hybrid's `tokens != 0` bail is **deleted** — it fired after the fact, and
  site 8 bypassed it entirely by reaching the session directly.

**P3.4's premise was also wrong.** "Remove the block-count parameter (it only
ever takes 0)" is false: the clean-turn re-prefill passes `seal_block_from` and
the glue-gap rollback passes the pre-reservation index. Both are legitimate —
neither leaves state describing tokens the K/V no longer holds — so the parameter
stays, documented at `batched_inference.rs`'s `truncate_sequence_to_blocks`.

### 11a.2 Phase 5a is NOT independently landable — 5a and 5b landed together

§8 says "5a is independently landable and independently valuable". It is not.
The coupling runs through the seal, and the failure is silent:

1. `assemble_pieces` emitted `markers.user_start` before **every**
   `Sealed::Turn`, unconditionally.
2. The dialogue seal builds the turn's layout from the submitted
   `user_content_start`.
3. `from_flat_grid` bakes a **real** leading `UserStart` when that is `> 0`.

P5a folds the opener into the turn's own prefill, which forces (2) non-zero,
which triggers (3) — and (1) would still fire. Every subsequently-projected turn
would then carry **two** openers, shifting every boundary after it, reading
perfectly.

**They landed as one change**, and the bake is **central**: it lives in
`Sequence::submit_prefill_unit`, the single funnel all three turn-producing
paths route through (dialogue, prefilled calibration, inserted). A path that
baked its opener and not its closer, or one path baking while another did not,
is exactly the silent-divergence this centralisation forecloses.

Two consequences worth naming:

- **The compression turn had to bake its own head and tail** (P5b.7 predicted
  this exactly). It builds its grid directly in the scheduler and carried the
  comment *"No leading `no_think` / `user_start` head: those are live
  `Generated` segments the assembler re-emits"* — which stopped being true. Left
  alone it would have projected with no opener at all.
- **`assemble_pieces` lost its `markers` and `turn_no_think` parameters.** They
  became dead the moment turns owned their boundaries, and removing them rather
  than silencing them is what makes "the assembler no longer decides boundaries"
  visible in the signature. `materialize_conversation` lost `markers` for the
  same reason.

### 11a.2b P5c was smaller than it looked — and two thirds of it is unused

§4.7d treats the three remaining `Generated` producers as comparable work. They
are not:

- **`TreeGlue` and `member_glue` are not used by the live schema at all.**
  `grep glue zend/src/prompts/projection.yaml` returns nothing. They are
  capabilities with no production caller, so sealing them is speculative work
  against an unexercised path.
- **`is_template` sections were a two-line change**, because a template section
  already carries its resolved dialect text in `SectionSchema::content` — the
  same field a content section seals from. The ingest loop skipped them with an
  explicit `continue`; removing it, and making `push_section_segment` always
  emit `Sealed`, is the whole of it. `push_section_segment` lost its branch and
  its panic (a template with no pre-tokenised tokens is no longer a way to fail).

So the live system's only glue producer is sealed, and the two unused ones are
left as they are — but they can no longer fail *silently*: `can_gap_fill` makes
`reserve_glue_island` refuse with a message naming the island, so enabling
either on this lineage stops the wave with an explanation rather than producing
quiet nonsense.

The approximation this accepts is the one §4.7d predicted: a template's K/V is
now computed under the ingest prefix rather than the runtime one. For
`depends_on` templates whose collection may or may not have materialised, that
is exactly the "approximation-rich prefix" the collection path already concedes.

### 11a.3 The hybrid is not reachable from the conversation layer (unlisted)

The 142-item list assumes zend can already load the hybrid. It cannot — or
rather, could not: `candle-conversation::models` had no `ModelArch` variant, no
`Model` preset and no builder arm for this lineage. `HybridBatched` existed only
behind candle-transformers' own `#[ignore]`d gates.

This is a prerequisite for the switch that appears nowhere in §11, and §0's
survey missed it by never asking whether `Model::` had an entry.

**Landed** (it blocks P9 and the switch itself): `ModelArch::Qwen35Hybrid`, the
`Model::Qwen36_35B_A3B_Q4` preset (`models/qwen36_moe.rs`), the builder arm, and
the `qwen35moe` GGUF arch-string mapping. Compile-checked only — loading it needs
the 22 GB checkpoint, so **the first person to run P9 is also the first to
exercise this loader**.

### 11a.3b P6.1's premise is false — the generator never prefills a branch prefix

§4.6 / P6.1 says the branch checkpoint "is a by-product of a prefill the
generator already runs — do not add a second pass." It is not, and the reason is
this document's own subject appearing one level up.

The section ingest **Arc-injects** the prefix and prefills only the section's own
content (`conversation.rs:441`, `:706` — *"every section in `prefix_section_ids`
is Arc-injected onto the scratch slot before this section's prefill"*). That is
the right design for K/V: the prefix's K/V already exists, so copying it is free
and the forward attends to real preceding context.

A recurrence cannot be Arc-injected. So after sealing a variant, the fork's
recurrent state covers **only that section's tokens**, computed under a K/V
prefix the recurrence never processed — KV-without-state, in the prompt builder,
exactly the mismatch §1 describes for conversations.

The consequence for P6 is not a detail. There is no existing prefill to harvest;
a branch checkpoint needs a **dedicated pass** that runs the full ordered prefix
through the model for each leaf. At a five-binary-plus-one-ternary tree that is
96 full-prompt prefills at build time — affordable as a one-off, but it is new
work with a new cost, not a by-product. P6.2's "build-time accelerator" (resume
the walk from the deepest shared ancestor) stops being an optimisation and
becomes the thing that makes the pass affordable at all.

**Not implemented**: it needs the model both to run and to validate, and getting
it wrong produces a conversation that starts with a *plausible* prompt state
rather than an obviously-empty one — the failure mode that reads fine.

### 11a.3c P8.1 and P10.3 were implementable and were still not done

Both could have been written blind. Neither should be, and the reason is the
same in each case: the change is only *correct* if a measurement or a gate says
so, and running either needs the checkpoint.

- **P8.1** (pin the export buffer, overlap the D2H) is guarded by its own phase
  condition — "only if seal latency measures badly". The measurement has not
  been taken. Landing async CUDA stream plumbing on the seal path, unmeasured
  and unrun, trades a cost nobody has shown is real for a class of bug (a race
  between the staging copy and the seal's own writes) that does not reproduce on
  CPU and would surface as intermittently wrong resumed state.
- **P10.3** (spec-decode on forks) means reworking deepseek4's verify path,
  which carries its own compressor and gallery rollback. It is a working,
  GPU-only, performance-critical feature with no CPU-runnable gate. The capability
  refusal (§11a.1) already makes the hybrid safe there; the rework buys deepseek4
  nothing it does not already have.

Recorded because "not done" and "not done *yet*, deliberately" are different
states, and the second one should not be re-litigated from scratch.

### 11a.4 `head_dim / 4` is a whole-word rotation at head_dim 256

P5e.3 derives the fold shift as `head_dim / 4`. At 128 that is 32 bits — a
half-word, which mixes bits *within* each u64. At 256 it is 64 bits, exactly one
word, so the stagger only permutes words and never mixes within them.

Found because a first draft of the stagger test used word-periodic data and the
layers cancelled anyway. That is weaker decorrelation than the measured 128-bit
case, and it feeds directly into P10.1's re-derivation: the question is not only
*which layers* the groups take but whether the shift should avoid word alignment.
The tests document it at `wide_sig.rs`'s `the_fold_staggers_at_head_dim_256`.

### 11a.5 Deviations of detail

- **P2's disposition is decided at the seal branch, not at `finalize_view`.**
  The doc has `finalize_view` move the state; but whether the turn keeps its
  decoded blocks is not known until the clean-reprefill branch has run. The view's
  state is therefore left under the view id (`finalize_view` frees the session
  slot but not the model's map entry) and disposed of at the branch — released on
  the clean re-prefill, moved everywhere else. Deciding it eagerly *is* the
  `<think>` skew.
- **P4.2 resolved as predicted**: the hook returns `Vec<ExportedLayerState>` +
  hash, and the scheduler assembles `SnapshotPayload`.
- **P7's rejections log at WARN with distinguishable reasons**, as specified —
  unreadable / model-carries-none / hash-or-geometry-mismatch, each naming that
  the conversation will "read fluently and have forgotten".
- **The `seeded` flag is consumed on the first wave regardless of offset**, not
  only at `offset == 0`. A flag that outlived its one wave would suppress a later
  genuine reset — defect 6 wearing the fix's clothes.
- **P5e.5/5a resolved without a record change.** The fold emits exactly three
  layer-groups — `FoldParams::group_sizes` is a `[usize; 3]`, so the count is
  pinned by the type — which makes `heads_per_group = n_heads / 3` **derivable
  from the signature itself**: 12/3 = 4 on Qwen3-30B, 6/3 = 2 on the hybrid. No
  new field, no version bump, and no churn across 41 construction sites. All
  four `HEADS_PER_GROUP` readers (`scan.rs`, `packed.rs`, `gpu.rs`, and the
  un-aliased copy in `zend/examples/provenance_layers.rs`) now derive it; the
  example's silent group-drop became an assert.
- **P5e.6–8 done, but the check is narrower than §4.8 specifies — on purpose.**
  The record is now `WQS5` and carries the fold (`encode_wide_sigs_with`); the
  scheduler publishes this process's fold at construction from the model's own
  geometry; the two substrate gallery reads go through
  `decode_wide_sigs_for_scoring`, which refuses a mismatch with a WARN.

  What it compares is **only the group sizes**. Everything else in a fold is
  recoverable from the signature's own shape — `heads_per_layer` is `n_heads / 3`,
  `head_dim` is `wph × 64`, `shift` follows — and the scorer already derives
  those per signature, so a shape difference is *handled* rather than being a
  mismatch to refuse. The group sizes are the one parameter the shape cannot
  reveal (they depend on the capture-layer count) and therefore the one case
  nothing else catches.

  This was found the hard way: checking the shape-derived fields too made every
  geometry constructed in a test process contaminate the next, and two substrate
  tests went red. That is not merely a test artefact — it is the same coupling
  in production, where a check that fires on shape would refuse comparisons that
  are correct.
- **`ModelCoreProperties::provenance_capture_layers`** is new and needed for the
  derivation: the fold groups `[n − 2, 1, 1]` over layers that actually have a Q,
  which is `num_layers` on a uniform stack and the **attention** count on a
  hybrid. Passing transformer depth would size the lower group for 30 layers that
  contribute nothing.
- **Both halves of the boundary bake exist, are tested, and are now used.**
  `from_flat_grid_with_tail` reserves the closing `<|im_end|>` the way
  `user_content_start > 0` already reserved the opener, and a zero reservation
  is asserted byte-identical to the ethereal form.
- **Stream equivalence is asserted directly** rather than by pinning the old
  island shape: `baked_boundaries_reproduce_the_spine_emitted_stream_exactly`
  builds the reference stream the spine used to emit (`US body AE` per turn,
  with the `AE ++ US` islands between) and asserts the baked layouts' `realize()`
  walk reproduces it **token for token**. Equality, not tolerance — §4.7a proves
  the split is exact. A companion test covers the `/no_think` rider that §4.7a's
  ownership table omits.
- **The `no_think` switch is now stronger, not weaker.** §4.7b worried that
  baking would freeze a deliberately dynamic decision. Baking it into the turn
  that carries it *removes* the leak the live path guarded against — a past
  suppressed turn putting a stale switch on a later thinking-on turn cannot
  happen when each turn's grid holds its own.

### 11a.6 The fold's bits and its stamp came from different places (P5e.1a)

P5e.1a says, in as many words: *"Thread it through **both** gather paths — they
are separate code and either can be missed."* Both were missed, and the result
was worse than either miss alone.

`Scheduler::new` derived the fold from the model's geometry and published it;
the seal stamped each `WQS5` record with that derived value; and both capture
paths — the GPU fast path's `assemble_folded_prov_sigs` and the CPU R16
fallback — still called `fold_provenance`, which is `fold_provenance_with(raw,
FoldParams::locked())`. `fold_provenance_checked`, written for exactly this and
described in §4.8 as the thing that refuses an unfillable fold, had **no
production caller at all**.

So a record's header described a fold its bytes had not been produced under.
On Qwen3-30B the two agree (`derive(4, 48, 128) == locked()`), which is why
nothing showed. On the hybrid the bits would be folded `[46, 1, 1]` at
4 heads/layer over a 10-layer, 2-head stack — groups 1 and 2 all zero — while
the header claimed `[8, 1, 1]` at 2 heads/layer. The mismatch check would then
*pass*, because it compares the stamp against this process's fold and both are
the derived one, and the scorer would run over two thirds of nothing.

**What landed.** The fold is a field on the `Scheduler`, set once at
construction from the model's geometry, and it is the value used for *both* the
fold and the stamp. One source, so they cannot drift. Both capture paths call
`fold_provenance_checked` and store nothing when the fold cannot fill all three
groups, warning once — an all-zero group is not a weak signature, it is a
scorer input that agrees with everything.

A second miss came out with it: the compression/summary path persisted its
node signature with `encode_wide_sigs` — **unstamped**. An unstamped record
reads back with group sizes `0`, which `decode_wide_sigs_checked` treats as
"not stated" and scores without checking, so summary nodes bypassed the fold
check entirely. They stamp now.

The general lesson is the one §0.4 already states and this document's own TODO
anticipated: a checked variant with no caller is indistinguishable from no
check, and "threaded through" is a claim about call sites that only a grep
settles.

### 11a.7 A cached checkpoint could not be opened while the hub was unreachable

Not a design point — a live failure, hit on the first tier-3 run. The test sat
for eighteen minutes on twenty seconds of CPU with two sockets in `CloseWait`
and the model never opened.

`ModelBuilder::download_or_fail` called `Api::get`, which consults the local
cache only *after* asking the hub which revision it should be holding. These
are pinned files — one filename in one repo, whose exact length the spec
records — so a cache hit needs no confirmation, and asking anyway makes every
load depend on the network. The same shape was in the test helper
`hf_get_repo`, where every caller pins an explicit revision.

Both now check the cache first and go to the network only on a miss. This is a
daemon startup property, not a test convenience: a workstation with the
checkpoint on disk should not fail to start because the hub is unreachable.

### 11a.8 A view that borrows no K/V reads perfectly

`BatchedInferenceSession::create_view_sequence` took `visible_block_ranges` and
treated an **empty slice** as "borrow nothing", returning a zero-block view
without complaint. "Empty means every block" is the *scheduler wrapper's*
convention — it expands to `[(0, total_blocks)]` before calling down — and the
two read identically at the call site.

A view carved that way decodes at its parent's position with an empty K/V. It
produced fluent, grammatical, entirely unrelated text, and the tier-2
continuation gate (T7.6) read that as a fork defect twice before the carve
itself was checked. The recurrent state was being copied correctly the whole
time; T1.6 passed throughout, because its claims are about state and the state
was right.

The empty slice is now an error naming the fix, and the tier-2 gates carve
through one helper that asks for the parent's whole block range and asserts it
got it. Worth recording because it is this document's failure signature
appearing in the *test harness*: the instrument reported a defect in the thing
it was measuring, and the report was fluent.

### 11a.9 `thinking(false)` does not suppress thinking

Observed while reading the P9.3 output, where the resumed conversation emitted
a real `<think>` block and the pre-restart one had emitted an empty one. It is
not a fork or resume defect. `ModelBuilder` has two levers and on this
configuration neither fires:

- `format_system_prompt` prepends `/no_think` to the **system prompt**, and
  `conversation.rs` records — correctly — that Qwen3 honours the switch only
  from the user turn, which is why a suppressed turn now bakes it into its own
  grid via `turn_head_tokens`.
- the non-thinking sampling params are applied only when the caller has not set
  sampling explicitly, and the tier-3 harness sets `argmax`.

So suppression was inert in both engines and the model simply chose
differently. Left as it is: making the builder's flag reach the per-turn
selector changes behaviour for every model and every conversation, and there is
no gate here for that. Recorded, and noted at the call site so the `<think>`
blocks in these tests are not read as a defect. The comment in
`submit_prefill_unit` that still described the deleted `no_think_current` glue
was corrected.

### 11a.10 T5b.5 is not implementable as written

*"With boundary K/V present, a `can_gap_fill: true` model produces identical
output whether it injects or regenerates — proving `Option` is a hit/miss, not
a mode."*

The assertion needs the same model, on the same turn, to take both branches.
After P5b a turn's `TurnLayout` is built at seal time and stored on the turn;
its `Glue` segments carry `kv: Some(span)` because the boundary really is in the
K/V. There is no way to make that turn regenerate instead — and the two ways to
add one are both prohibited:

- a switch saying "regenerate even though you have it" is optionality-as-a-mode,
  which is what the test exists to disprove; and
- a mutation path that rewrites a sealed layout's `kv` to `None` would be a
  test-only API on the seal path.

**What the property rests on instead.** `TurnSegment::is_real()` is a plain
`kv.is_some()`, and both values occur naturally for reasons that have nothing to
do with capability: a `<think>` block whose K/V was deliberately dropped is
`None`, a sealed boundary is `Some`. Nothing reads `can_gap_fill` to decide
which to write. The capability is read in exactly one place —
`reserve_glue_island` — and both of its outcomes are already gated: the hybrid
refuses (the feasibility gate, T5c.1), and Qwen3-30B does not.

What would settle the remaining question is not a test but a comparison across
the bake: the same Qwen3-30B conversation, same seed, before and after P5b.
That belongs with P10.4's ablation, which already has to measure the two baked
approximations, and it is recorded there rather than left as an unwritable test.

### 11a.11 P6 landed, and §4.6's cross-product is the wrong shape

P6 is implemented: the per-leaf prefill pass, its persistence, and the
cold-start restore. Two things about it are **not** what §4.6 specifies, and
both were settled by measurement rather than argument.

**(1) A conversation computes its own branch, not the cross-product.** §4.6
says to pre-seal every leaf the way the K/V tree does. The first working
version did exactly that and the tier-2 gate reported **200 checkpoints
computed** on a single conversation open — the live schema's tree is
`no_think(2) × persona(2) × reasoning_stance(2) × thinking_effort(5) ×
response_length(5)`, not the "five-binary-plus-one-ternary = 96" §4.6 estimates.
A conversation uses exactly one of those 200. Eagerly computing the rest is not
warm-start, it is 199 full-prompt prefills of latency in front of the first
turn, for branches that may never be selected.

§4.6's own conclusion already points the other way — *"what is needed at runtime
is only the leaves"* — and the reason it did not follow through is that the K/V
tree really does pre-seal the cross-product, so the symmetry looked right.
It is not symmetric: sealing a variant's K/V is one section's prefill, while a
branch checkpoint is the *whole prompt* every time, because a recurrence cannot
be Arc-injected (§11a.3b).

So a branch is computed the first time some conversation selects it and
persisted under its content prefix. Cost: one prefill per distinct branch ever
used, paid once across every conversation and every restart. **This also
retires P6.2**: resume-from-the-deepest-shared-ancestor exists to make a
cross-product affordable, and there is no cross-product. The enumeration and
addressing survives as `SectionTree::branch_prefix_ids` (with T6.1/T6.2),
because that is what *names* a branch and the pass uses it to derive one from a
selection. The enumeration that went with it does not: a
`leaf_selections` that returned every reachable assignment had no caller once
the cross-product went, and a capability with no production caller is the exact
thing §11a.2b criticises about `TreeGlue`.

**(2) The checkpoint is keyed by content prefix, not by `pack`.** P6.3 says to
key on `SectionTree::pack(selection, dims.len())`. That names a branch only
within one build of one schema: `SectionId`s are assigned in declaration order,
so editing a section's text — or inserting one above it — leaves every pack key
pointing at a branch whose tokens have changed, and the restore seeds a
conversation with the state of a prompt it is not running. Keying on the
cumulative `ContentChain` prefix, which is what the sealed K/V is already
addressed by, makes a checkpoint and its K/V go stale together by construction.

**What the gate caught on its first run.** T6.3 asserted the install path and
found **200 computed, zero installed**. The write is fire-and-forget onto the
persistence thread and the restore read the record back immediately, before it
had landed — so every conversation did the whole pass and then started from
zero, warning once at a level nobody reads. Reading back a value computed
moments earlier was the mistake; the default branch's payload is now carried in
memory to the install, and disk is what the *next* process reads. This is the
same shape as every other defect in this document: it cost real work, produced
no error, and the conversation read perfectly.

### 11a.11a How P6 is put together

The pieces, since none of them is where §11 says:

- **`SectionTree::branch_prefix_ids`** — the ordered sealed sections a branch is
  built from, with placeholder nodes contributing their own anchor rather than
  the collection's runtime top-k (§4.6(2)). `Sequence::prompt_branch` derives the
  branch from the conversation's own selection through it, rather than trusting
  the ingest walk's layout, so a conversation opened on a non-default assignment
  names the branch it will actually run. T6.1/T6.2 cover it, including that an
  out-of-scope gated dim resolves to one branch however it is set.
- **`SchedulerRequest::BranchCheckpointPass`** — prefills an ordered token
  stream on a throwaway slot and exports the state. Deliberately *not* routed
  through the section-ingest batcher: that path exists to share forwards across
  many concurrent section prefills and finishes by sealing K/V, and this wants
  neither. It drives `forward_wave` directly in `max_prefill_pass_tokens`
  chunks, for the same reason `build_section_batch` bounds its own budget — one
  forward over a whole prompt is an activation spike large enough to page.
- **`BranchCheckpointPayload`** — rides `RecordType::Snapshot` under a different
  stream id. Nothing in the single-tail machinery needed changing: accounting,
  the recovery walk's location map and the compactor's carry-forward all key on
  `header.stream_id` and never decode a payload. Two payload shapes under one
  record type is not an overload, because the stream id *is* the identity of
  what the state belongs to and a reader computes it before asking.
- **`SchedulerRequest::InstallRecurrentState`** — the restore. It runs *after*
  priming rather than inside `create_sequence` like the timeline-snapshot
  restore, because a branch checkpoint describes the state after the prompt and
  the prompt's K/V has to be on the slot first.

**The payloads needed a magic byte, and the test that found out is worth
keeping.** Without one they are wire-*compatible*: `(version, timeline, turn,
schedule, n_layers)` and `(version, prefix_lo, prefix_hi, schedule, n_layers)`
are the same widths in the same order, so a conversation snapshot decodes as a
branch checkpoint cleanly, every field landing somewhere plausible. The test
asserting the refusal failed, which is how `BRCK` came to exist.

### 11a.12 What the state costs — measured

Six sealed turns on the 35B, via `recurrent_state_cost_is_measured_not_assumed`.
Numbers, not estimates, and each one closes or re-opens an item.

| | Measured | Verdict |
|---|---|---|
| **T8.1** seal export | 39.8 ms/seal, **4.06 %** of turn wall | **P8 stays closed** |
| **P10.5** snapshot write | **62.8 MiB** per turn | **re-opened** — see below |
| **P10.6** fork traffic | 18 forks / 6 turns, **~188 MiB per turn** | **re-opened** — 3× the prediction |

**P8 is closed, with evidence rather than by deferral.** §11a.3c declined to pin
the export buffer and overlap the D2H because the phase is conditional on a
measurement nobody had taken, and landing async stream plumbing on the seal path
risks a race between the staging copy and the seal's own writes — intermittently
wrong resumed state, not reproducible on CPU. At 4 % of turn wall the saving is
bounded by 4 %, and that is not worth the class of bug. The gate now asserts the
threshold (10 %) rather than the conclusion, so the day the ratio moves, the
decision re-opens by itself.

**P10.5 deserves attention.** 62.8 MiB per turn is exactly the ~63 MiB §7
predicted, which is the good news; the bad news is that it is per turn, to the
redo log, forever, and it is the single largest recurring write in the system. A
conversation at one turn per second sustains ~63 MB/s of snapshot alone. bf16
storage halves it for a quality question §7 already framed and P10.5 already
owns. This is now a measured cost rather than a hypothetical one.

**P10.6 found three forks per turn, not one.** §4.4 and P10.6 both reason about
"a ~63 MiB device copy on every dialogue turn". There are three, so ~188 MiB per
turn moves device-to-device. P10.6 asks precisely the right follow-up already —
*"whether a view's fork can be deferred to the first layer that actually
advances, since a turn that reprojects before decoding anything has paid for a
copy nothing read"* — and the multiplier makes it three times more worth asking.
The next step is to find out what the other two carves are: a turn takes one view,
so either reprojection carves again or the seal path does.

**P10.1, P10.2 and P10.4 are experiments, not code.** Each needs a labelled
retrieval corpus and an analysis pass, not an implementation: which layers carry
identity on a 10-attention-layer stack (P10.1), whether attention-only capture
retains retrieval quality (P10.2), and the zero-glue/fork-glue ablation together
with the two baked approximations and the pre/post-bake comparison §11a.10 moved
here (P10.4). What this campaign changed is that they are now *runnable*: the
fold derives correctly on the hybrid, and the layer-attribution harness no longer
silently scores an empty projection when the fold shape moves (§11a.6, P5e.5a).

### 11a.13 Gaps found by reading this document against the code

Five, found by cross-referencing §4/§9/§10 against what shipped. Three were
real defects, one was a decision whose premise had gone, one was a question
nobody had answered.

**1. `turn_index` was logged on resume, never validated.** The worst of them,
because §4.1 *depends* on it. The seal enqueues the snapshot **before** the
turn's `Tokens` record, justified as: a torn shutdown can then leave a snapshot
for a turn whose records never landed — "handled" — but never the reverse. The
handling is §4.3's *"reload discards a snapshot newer than the last recovered
turn"*, and `restore_recurrent_state` only ever passed `payload.turn_index` to
`tracing::`. So the write ordering deliberately produced a tear it could not
survive: resume installed a state one turn **ahead** of its K/V — state without
KV, §7.8 defect 2, and fluent. Now rejected with a distinguishable WARN, behind
`snapshot_within_recovered_history` so the boundary is unit-testable (T7.3).
This is also what makes §10 decision 3 (leave a turn-tombstoned snapshot alone)
work at all.

**2. Branch checkpoints had no reclaim path.** Introduced by this campaign, not
by the design. Turn snapshots have three: supersede by header key, timeline
tombstone, distillation. A checkpoint is keyed by *content*, so a prompt edit
supersedes nothing and orphans the old branch, and no timeline tombstone names
it — the log grew by one 63 MiB orphan per prompt edit, carried forward verbatim
by every compaction, forever.

The fix names what a checkpoint actually is. It is a **cache**: a pure function
of a prompt still on disk, so losing it costs one prefill, where losing a
conversation snapshot loses history nothing can recompute. That distinction now
lives in the type — `RecordType::BranchCheckpoint = 21`, its own substrate index
— because the compactor reads headers, never payloads, and had no way to tell
them apart. Compaction keeps the newest `MAX_BRANCH_CHECKPOINTS`; maintenance
relocates them under their own type so a cache is never re-typed as durable
state.

**3. §10 decision 4's branch-checkpoint half is revised, not implemented.** It
says branch checkpoints store bf16, on the reasoning that *"the cost is fixed and
paid up front rather than growing with use"* — 96 leaves ≈ 6 GB. Two things
since: the pass computes one branch per conversation rather than the
cross-product (§11a.11), and compaction now caps them. The worst case is
`MAX_BRANCH_CHECKPOINTS × 63 MiB`, bounded and small, so the premise for
measuring bf16 *before* landing is gone.

What is left is the ordinary version of the question — is lossy stored state
acceptable — and §7 already answers how to settle it: *"a measured quality gate,
not an assumption."* That gate is P10.5. Landing bf16 now would be shipping a
quality change ahead of its measurement, which is precisely the trade §11a.3c
declined for P8.1. Both payloads are F32; the per-layer dtype tag is still the
extension point, and it is honestly documented as F32-only today rather than
described as a capability that does not exist.

**4. T3.6 was never written**, and §10 decision 9 made the bake conditional on
it: *"assume fixed once sealed, and add the test. If the test fails, fall back
to leaving `NoThink` ethereal."* §11a.5 argues the property is now stronger,
which is an argument, and the decision asked for a test because baking freezes
an assumption. Written: the flag is derived from the segments (no second copy to
drift), survives the persistence round-trip, and two turns sealed under
different dial settings keep their own answers.

**5. §9 Q6 had no recorded answer.** *"Confirm no other model in the tree carries
recurrent state that should ride this."* The answer is on
`carries_recurrent_state` now, because the confirmation is what the doc comment
should have said in the first place: per-sequence state outside the K/V is not
the criterion — **irrecoverability** is. `latent_moe`'s engine carries a
per-sequence compressor and provenance gallery and answers `false` correctly,
because both are derived from a corpus that is already durable. A delta-rule
matrix has no such source; it is the only record of the tokens that built it.

### 11a.14 Gaps found by a code review of the finished work

The behavioural catalogue proved the *design* holds. A full review of the diff
found ten defects the catalogue's oracles could not see, six of them in this
work and three of those introduced by its own fixes. Recorded because the
pattern is more instructive than any single bug: **every one of them was a
place where two derivations of the same fact were allowed to disagree.**

1. **A tracked "state is still pristine" flag** gated the first-turn branch
   re-key, and only the dialogue path cleared it — so an ingest or a splice
   advanced the state with the window still open, and the next dialed turn
   installed the bare prompt checkpoint over it. Now DERIVED: the flag says
   only *born with a checkpoint*, and the timeline's turn count says whether
   anything has advanced it. Every path that advances state lands a turn, so
   nobody has to remember.
2. **The provenance palette count was derived on one side and constant on the
   other.** `prov_n_palette` computed 8 bands at `head_dim` 256 while the arena
   stores 4 and the kernel's contract is `head_dim / N_PALETTE`; the fast path
   re-activated and captured half of every signature, dim-permuted, no longer
   bit-identical to the CPU fold it is documented to match. The band count is a
   property of the R16 arena, not a free choice: it reports `N_PALETTE`, and a
   256-wide head correctly declines to the (slow, correct) CPU path until the
   arena and kernel genuinely band that wide.
3. **The fold-group divisor was swept everywhere but the scorer.**
   `resolver.rs` still divided by the locked `PROV_HEADS_PER_LAYER`, reading 1
   group for the hybrid's 6-head signatures — collapsing three-group late
   fusion to a single scan and making `layer_weights` address groups never
   scanned. Now `heads_per_group`, the same derivation as the capture side.
4. **The splice catch-up replayed adopted tokens against an empty context.**
   The scratch slot held the parent's state but none of its K/V, so the
   attention layers fed the recurrent layers outputs no forward ever produced.
   The scratch now BORROWS the conversation's leading blocks (`prefix_blocks`,
   the extent before the adopted span), giving the replay the left context a
   real append would have had — without double-counting K/V that is already
   spliced in.
5. **The post-decode tail advanced the wrong state.** Prefilled before the
   view-state disposition, it was absorbed twice on the clean-re-prefill path
   and lost on the move path — the same skew the disposition comment describes
   for `<think>` and deliberately avoids, one call earlier. It now runs after
   the disposition, so the state that absorbs the tail is the one being sealed.
6. **Boundary ownership moved into the grid, and history was not asked.** Turns
   sealed before the move carry ethereal markers that `realize` drops, and the
   assembler had stopped emitting them — so a resumed pre-existing workspace
   projected as one unbroken run with no role markers anywhere. The assembler
   now asks each turn's own layout (`TurnLayout::bakes_own_boundaries`) and
   supplies the framing for exactly the turns that lack it.

Also fixed: the write-half of a forked recurrent state was zeroed
(invariant 6 — ~63 MiB of memset per fork, ~3 forks/turn) and is now
`uninit`; summariser scratch slots inherited the conversation's recurrent
state and belief through the `create_sequence` funnel (`StateSeed::Neutral`
now says a timeline can be an *address* rather than a history to continue);
the branch checkpoint was read and decoded twice per conversation open; and
`clear_walker_state` cleared the branch-checkpoint index but not the
recurrent-snapshot one, which compaction had just been taught to shrink.

## 12. References

- `docs/qwen35_qwen38_models.md` §5 (turn-seal snapshots, the original design),
  §7.8 (the three invisible defects), §7.16 (F32 accumulation), §8 risk 1.
- `docs/tool_provenance_distillation.md` — the strip-KV-keep-sigs precedent.
- `docs/kv_tier_migration.md` §5.6-5.7, §16.12 — resume and the redo log.
