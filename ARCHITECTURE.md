# Architecture

This document describes the system architecture of this repository: an
**unbounded-context LLM inference engine** built as a heavily modified fork of
[Hugging Face Candle](https://github.com/huggingface/candle). It is not a
description of upstream Candle. Upstream Candle is a tensor library; this fork
is a persistent inference server that owns request scheduling, a paged and
quantized KV cache spanning three storage tiers, a live provenance-retrieval
index, an MoE expert-streaming pipeline, and an on-disk conversation substrate
— all built on top of Candle's tensor/op layer. Read this document to
understand how those pieces fit together and where to find each one in code.

The canonical technical report is [`docs/unbounded_agents.md`](docs/unbounded_agents.md)
("One Card, One Stack"); the root [`README.md`](README.md) is a condensed
summary. This document is the map between that theory and the actual crates,
modules, and files.

---

## 1. The thesis: unbounded context is a memory-hierarchy problem

Persistent agentic systems (a coding assistant with institutional memory, an
NPC with a lifetime of narrative history) need context that grows without
bound across sessions. Standard full attention cannot do this: every token
participates in every subsequent generation step with equal structural
weight, so under any finite-precision arithmetic, accumulated numerical
error grows as **O(N)** with context depth N. More VRAM defers the wall; it
does not remove it.

This system's answer is architectural, not a bigger compression ratio:
**decouple the number of tokens attending any generation step from the total
number of tokens in context.** A fixed-size working set is assembled per
step by **provenance-selected attention** — a retrieval mechanism, not a
sequential window — over a **three-tier paged KV cache** (GPU hot → RAM warm
→ NVMe cold). Because the working set size is bounded by a hardware constant
independent of N, `docs/unbounded_agents.md` §11.2 proves the expected
numerical error per generation step is **O(1)**, not O(N):

```
E[Σ ε(t)] ≤ ε_hot + W_warm_max · ε_warm + O(1/N) = O(1)
```

Hot-tier tokens are prefill-refreshed (zero decode drift, quantization error
bounded by the selection kernel's threshold); the warm tier's contribution
is capped by the fixed selection budget regardless of how large the warm
corpus grows; any specific cold-tier token's probability of entering the
working set at a given step shrinks as `O(1/N)`. This is why the
highest-compression formats (C7–C9, §4) are safe on cold/V-cache blocks —
the theorem makes their contribution asymptotically negligible — and why
this is not "just" a compression scheme: full attention over a maximally
compressed history still accumulates O(N) error, since removing the error
term requires removing tokens from the working set, not shrinking them.

Four subsystems implement this thesis:

1. **Provenance-selected attention** (§5) — Q vectors captured live during
   decode; a Binary Directional Provenance (BDP) scan ranks all KV chunks in
   3–10 ms regardless of corpus size.
2. **Three-tier paged KV cache** (§4) — GPU hot / CPU RAM warm / NVMe cold,
   32-token block granularity, async migration.
3. **Adaptive per-block KV quantization** (§4) — 11 compression levels
   (`compression_level` 0–10, informally "C0–C9" in the docs), K/V format
   chosen independently per block from cosine-distance thresholds.
4. **Markov expert prediction + wave-batched MoE** (§6) — prior-layer routing
   predicts current-layer expert loads; many sessions step through layers
   together so PCIe expert loads amortise across the whole batch.

---

## 2. The crate stack

```
zend                    OpenAI-compatible daemon: HTTP API, repo ingest
                         (repo_map / code_read), tools, web UI
       ↓
candle-conversation      conversation substrate, projection builder,
(~50 KLOC)               scheduler/wave engine, provenance retrieval,
                         persistence — this is where the "engine" lives
       ↓
candle-transformers      model impls, batched inference API,
                         MoE expert-streaming pipeline
       ↓
candle-nn                layers, VarBuilder, KV cache subsystem
                         (paging, arenas, adaptive quantization)
       ↓
candle-core              Tensor, Device, DType, op dispatch,
                         CPU/CUDA/Metal backends
       ↓
candle-kernels           720+ AOT-compiled CUDA kernels: paged-decode/,
                         paged-prefill/, paged-glue/, quantized/,
                         provenance/, simple/, sampling/
```

Each layer only calls down; `candle-kernels` never depends on anything above
it, and `zend` is the only crate that speaks HTTP. `candle-conversation` owns
everything that makes this an *engine* rather than a model-forward library:
the scheduler thread, the provenance index, the persistence layer, and the
projection/substrate system that assembles context windows. `candle-examples`
and integration tests sit alongside `candle-conversation` as consumers.

---

## 3. The request lifecycle

This is the single most useful section for understanding the system: what
happens, module by module, from an HTTP request to a sampled token landing on
disk.

```
HTTP POST /v1/chat/completions              zend/src/api/chat.rs
   ↓
ZendSession resolves/creates a Sequence     zend/src/session.rs
   against the Substrate + projection Schema
   ↓
Substrate append + projection Builder       substrate.rs; projection/
   (turns/sections appended; content         builder.rs, resolver.rs,
   resolved via ContentResolver)             project.rs
   ↓
SchedulerRequest over channel                scheduler/mod.rs
   (single background thread owns all GPU state)
   ↓
SCHEDULER LOOP — admission + wave forming    scheduler/run.rs
   drain_submissions → promote_new_prefills (VRAM-gated admission)
   form a wave: decode + prefill + glue row-groups for this quantum
   (continuous fair waves: decode sweeps every layer every wave; a
   large prefill/glue creeps through layers at a throttled rate, §6)
   ↓
FORWARD DISPATCH, per layer                  batched_*.rs (candle-transformers)
   paged_decode / paged_prefill / paged_glue attention kernels each
   fill their disjoint rows of one [total_tokens, hidden] buffer
   → shared o_proj → shared MoE grouped-GEMM pass (§6) → residual
   ↓
PROVENANCE CAPTURE, every decode step        candle-conversation/src/provenance/
   live Q vectors from the R16 KV format → folded WideQSig
   (raw_store.rs, wide_sig.rs), accumulated across the probe window
   ↓
RETRIEVAL / RESELECTION, per reprojection
   BDP scan of the probe vs. the resident gallery arena
     (provenance/gallery_arena/, candle-kernels/src/provenance/)
   → per-scope normalized scores (candle-conversation/src/normalization/)
   → belief update (provenance/gather.rs, belief.rs)
   → feeds the projection Builder for the NEXT reprojection
   ↓
TIER PREFETCH, async, overlapped with compute
   selected warm/cold blocks promoted toward VRAM hot tier via the
   migration kernel (candle-nn/src/kv_cache/chunked/migrate.rs),
   driven by persistence/transfer.rs, elevate.rs
   ↓
SAMPLING                                     scheduler/sample.rs
   fused single-launch kernel (temperature/top-k/top-p/DRY/…),
   candle-kernels/src/sampling/
   ↓
TURN SEAL, on turn completion                scheduler/mod.rs (seal path)
   prefill refresh over the completed turn — re-quantizes from clean,
   non-decode-drifted activations; per-block format re-selected
   (candle-nn/src/kv_cache/chunked/compress.rs)
   ↓
PERSISTENCE                                  candle-conversation/src/persistence/
   sealed chunks appended to the redo log (writer.rs, log_file.rs);
   hot→warm migration on the persistence thread (thread.rs);
   substrate metadata (streams, manifest) kept consistent
   ↓
Streaming response back to zend → HTTP client (SSE)
```

Two structural points are easy to miss reading the code piecewise.
**Retrieval is decode-time, not query-time-once**: the provenance scan runs
continuously as the model generates (`docs/attention_provenance.md` §5,
"Speculative Context Decode") — a probe session runs a few tokens ahead of
the kept decode session, its folded Q signatures drive the next scan, and the
next context window is assembled while the current one is still decoding;
probe tokens are discarded and never enter the KV cache. **Selection and
forward execution are decoupled by a full wave**: the scheduler never blocks
a forward pass on the CPU scan — the scan's result feeds the *next*
reprojection, which is what keeps the 3–10 ms scan off the GPU's critical
path.

---

## 4. The KV cache subsystem (`candle-nn/src/kv_cache/`)

The most actively developed part of the codebase. Two cache families exist:
`Cache`/`KvCache` (`cache.rs`) is the contiguous baseline used by
non-batched/simple paths; `ChunkedKvBacking` (`chunked/`) is the paged,
quantized, multi-tenant cache that the batched inference engine actually
runs on.

### 4.1 Chunking, arenas, GIDs

Every KV block is `CHUNK_SIZE = 32` tokens — a constant shared verbatim
between Rust (`chunked/types.rs`) and CUDA (`arena_table.cuh`), enforced by
the build so a mismatch cannot silently compile.

- **`Arena`** (`chunked/arena.rs`) — Float or Quantized backing, one of
  several `StoragePolicy` layouts, allocated in `TARGET_ARENA_BYTES = 16 MiB`
  slabs.
- **`ChunkGidPool` / `ChunkGid`** (`chunked/gid_pool.rs`) — a lock-free,
  refcounted slab allocator. `ChunkGid` is an RAII handle
  (`id = arena_idx * stride + chunk_idx`); allocation is O(1) (pop a Treiber
  free-list, else bump a high-water mark); the last `Drop` recycles the slot
  lock-free. `ResolvedArenaInfo` (`arena_table.rs`) resolves a GID to an
  absolute device pointer (`base_ptr + chunk_idx * chunk_byte_stride`) — the
  pattern every other paged subsystem in this codebase (paged-KV kernels,
  the gallery arena of §5.2) reuses.
- **`HeadGids`** (`chunked/head_gids.rs`) — per-head/palette chunk-GID
  collection for one sequence.
- **`ArenaTable` / `PerHeadTable`** (`arena_table.rs`) — the per-head format
  and palette index the attention kernel reads once per tile.
- **`SealedSequence` / `SealedChunk`** (`chunked/types.rs`) — the immutable,
  byte-stable representation of a completed sequence's KV; the unit
  persistence and tier migration operate on.

### 4.2 Adaptive per-block quantization (C0–C10)

`CompressionPolicy` (`chunked/compression_policy.rs`, backed by
`chunked/sampled_selection/params.rs`) carries a `compression_level: u8`
(0–10) plus per-model K/V error-threshold factors. `production_adaptive_candidates(level)`
returns ordered K and V candidate format lists; `compress.rs` evaluates each
candidate at seal time against a cosine-distance threshold and keeps the
smallest format that passes. Levels 0–1 are the `"quality"` tier, 2–4 are
`"sweet"`, 5–10 are `"compress"` (`PRODUCTION_LEVEL_TIER`). Thresholds are
**asymmetric between K and V** — K is channel-sensitive, V is token-sensitive
— tracked as separate `PRODUCTION_K_QREL_*` / `PRODUCTION_V_QREL_*` tables,
and are **per-model** (`LLAMA_KV_FACTORS`, `QWEN3_8B_KV_FACTORS`,
`QWEN3_MOE_KV_FACTORS`) because thresholds must be re-derived per model
family. **Attention sink protection**: positions 0–3 use a dedicated fine
scale (Q4_KS/Q8_KS) so attention-sink magnitude outliers cannot inflate the
global scale for the rest of a block. Format is selected **per head, not per
block** — one format per KV head, read once per attention tile and broadcast
across the warp.

### 4.3 The three tiers and migration

```
   GPU VRAM (hot)  ──async──▶  CPU RAM (warm)  ──async──▶  NVMe (cold)
   full-speed attn             pinned staging pool           redo log
```

- **Hot** — live `ChunkedKvBacking` arenas in VRAM.
- **Warm** — pageable CPU arenas driven by `candle-conversation/src/persistence/`
  (not the pinned `warm_pool.rs` originally designed in
  `docs/kv_tier_migration.md` — see the divergence note in `CLAUDE.md`). An
  evicted `SealedSequence` becomes `ArenaLocation::Cpu` with its GIDs
  re-pointed via `HeadGids::map_unique`; the original GPU GIDs drop,
  reclaiming VRAM through the allocator's RAII.
- **Cold** — the append-only redo log at `.substrate/substrate.log` (§8.3).
  Cold storage persists **KV cache blocks, not raw tokens** —
  `docs/unbounded_agents.md` §7 explains why: replaying prefill over stored
  tokens would not reproduce the same KV values the model attended with
  under provenance selection, raising the error floor above what the
  theorem assumes.

Migration is one primitive doing both directions: `kv_pack` (evict, gather
scattered arena chunks into a contiguous staging buffer) and `kv_unpack`
(load, scatter staging back into freshly-allocated chunks), implemented as a
single kernel (`candle-kernels/src/simple/kv_migrate.cu`,
`kv_migrate_copy` — one thread block per copy record, 16-byte-vectorised
where alignment allows) and orchestrated host-side by
`candle-nn/src/kv_cache/chunked/migrate.rs` (`MigrationPlan`,
`MigrationRecord`). The scheduler tick is one plan, one launch per direction
— independent of how many sequences are involved — on a **dedicated copy
CUDA stream** that overlaps with ongoing decode. Full design:
[`docs/kv_tier_migration.md`](docs/kv_tier_migration.md).

---

## 5. Provenance retrieval (`candle-conversation/src/provenance/`)

Retrieval replaces "attend to everything" with "attend to what a live
cognitive-state fingerprint says is relevant." Full design:
[`docs/attention_provenance.md`](docs/attention_provenance.md).

### 5.1 Capture

Every decode step's Q vectors are captured from the R16 KV format
(`raw_store.rs::extract_q_vector_r16`) and **folded** into a compact
`WideQSig` (`wide_sig.rs::fold_provenance`, 12 heads × 2 words = 192
bytes/token) — the "Binary Directional Provenance" signature, the packed
sign bits of Q itself (bit set ⇔ `Q[i] >= 0`). Retrieval is therefore a
**decode→decode `Q·Q` consensus**: a query token's folded signature is
compared against every gallery token's folded signature by sign agreement
(XNOR + popcount), not against K and with no PCA projection. At turn seal,
the turn's full signature window is
gathered (`gather_wide_sigs`, `scheduler/mod.rs`) and persisted on the
substrate as the `wide_q_sigs` blob (`substrate.rs::set_wide_q_sigs_blob`).

### 5.2 The paged VRAM gallery arena

Re-uploading every candidate turn's signatures on every scan was the
bottleneck; `provenance/gallery_arena/` fixes this by keeping folded records
**resident in VRAM** between reprojections, modelled directly on the
chunked-KV slab allocator (§4.1): fixed 6 KiB group-major pages (32 tokens
each), a Treiber free-list pool (`pool.rs`), persistent 16 MiB storage slabs
(`storage.rs`), and — like the paged-KV → CUDA interface — the scan kernel
receives **pre-resolved absolute device pointers** plus a `pos_map`
(logical token → `(page, offset)`), not an in-kernel block-index table. The
warm/cold tiers already exist independently (the substrate's `wide_q_sigs`
blob + its `decoded_wide_sig` `Arc` memo, and the redo log); the arena owns
only hot-tier VRAM residency and rebuilds an evicted turn from the `Arc` on
demand. Design: [`docs/paged_gallery_arena.md`](docs/paged_gallery_arena.md).

### 5.3 The scan kernels (`candle-kernels/src/provenance/`)

`bdp_scan.cu` is the scalar reference; `bdp_bmma.cu` is a 1-bit tensor-core
backend (`BMMA.88128.XOR.POPC`, sm_75–sm_89 only — Hopper/Blackwell dropped
the instruction), ~9.5 ms steady-state on the dev 4090; `bdp_imma.cu` is the
INT8 tensor-core backend (`mma.m16n8k32.s8`) and the production path on
Blackwell, projected ~6 ms on one RTX 5090. All three are verified
**bit-identical** by an adversarial parity test, sharing the leader/
runner-up vote logic in `bdp_vote.cuh`.

### 5.4 Scoring, normalization, and feeding selection

Raw dot-product scores are converted to a stable 0–1000 scale by
`candle-conversation/src/normalization/` — an asymmetric-EWMA "hit level" per
candidate, per *score-competition scope* (a turn group, a section
collection — never across scopes, never at the token-budget "layer" level),
so one selection threshold works uniformly across heterogeneous content.
Design: [`docs/provenance_score_normalization.md`](docs/provenance_score_normalization.md).
Normalized scores update an online per-tool/per-turn belief
(`provenance/gather.rs::belief_step`, `provenance/belief.rs::ToolBelief`)
that `projection::Builder`/`resolver.rs` reads on the *next* reprojection to
decide which turns, sections, and tools are in-window.

### 5.5 The immutable summary forest

History beyond raw turns is indexed at multiple resolutions by an
append-only 8-ary Merkle Mountain Range (`summary_tree/`, `tree/`) —
`SummaryOfTurns` leaves over one exchange, `SummaryOfSummaries` nodes over
exactly 8 same-level children. Nodes are immutable once created (no
AVL-style rotation, no `dirty` bit); the BDP scan drills from a coarse
"peak" node down into its immutable children to recover detail — the same
provenance-selected attention, rooted at multiple resolutions instead of one
global root. Design: [`docs/immutable_summary_forest.md`](docs/immutable_summary_forest.md).

---

## 6. The MoE expert pipeline and Markov prefetch

MoE inference (Qwen3-30B-A3B: 30B total, 3B active) cannot hold every expert
resident on a 16 GB card, so `candle-transformers/src/models/expert_lre/`
runs a background pipeline thread (`handle.rs`, `pipeline.rs`) owning a
fixed-size VRAM expert pool with `&mut self` (no lock on the hot path):
callers submit routed-expert requests over a channel; the pipeline
partitions them into cache hits (compute immediately) and misses (DMA-load
from host mmap, then compute), overlapping DMA with hit compute. Eviction is
a four-part policy (`mod.rs` doc comment) tuned to avoid eviction cascades.

**Markov expert prediction** (`expert_lre/transition.rs`) — an online,
self-learning transition matrix predicts a layer's expert routing from the
prior layer's observed routing (no offline calibration), 69% hit rate on
Qwen3-30B-A3B (`docs/markov_expert_prediction_eval.md`). The production
"Markov Wave" design runs two modes: a Bayesian prior + live per-session
predictor below the PCIe-saturation batch (≈256), and deterministic
prefetch-all-missing streaming above it, where prediction adds nothing once
bandwidth-bound. Eviction inside a wave is one deterministic rule: **always
evict the layer the wave just left** (`L-1`, wrapping) — the longest reuse
distance until the next full sweep.

**Wave-batched grouped GEMM** (`expert_lre/compute.rs`) steps every
in-flight session through a layer together, so an expert loaded over PCIe is
shared by every token routed to it that wave — PCIe cost amortises across
the batch instead of being paid per session. **Continuous fair waves**
(`docs/continuous_fair_waves.md`) decouples the *layer* traversal from the
*inference* wave: a decode cursor sweeps every layer every wave (its hot
experts stay continuously re-touched, never aging out of the LRU) while a
background prefill/glue cursor creeps through layers at a rate set by a
per-layer `decode_priority` (`Low`/`Normal`/`High`, the decode-to-prefill
airtime ratio). Where the two cursors coincide at a layer, their tokens
co-batch through one grouped GEMM.

---

## 7. The CUDA kernel layer (`candle-kernels/`)

Kernels are compiled **ahead-of-time**, not JIT-loaded at runtime.
`candle-kernels/build.rs` (shared logic in `build_utils.rs`) partitions
`src/*.cu`/`*.cuh` into named **archive groups** (`simple`, `quantized`,
`sampling`, `provenance`, `paged-decode`, `paged-prefill`, `paged-glue`, …),
computes a **SHA-256 hash per kernel and per group**, and only re-invokes
`nvcc` for groups whose hash changed — the rest come from cached `.a` static
archives. Each kernel builds with `-gencode=arch=compute_89,code=sm_89`
(native SASS, dev 4090) plus `-gencode=arch=compute_120,code=[sm_120,compute_120]`
(native SASS for Blackwell **and** embedded PTX as a forward-compat
fallback) — "PTX embedded at compile time" means embedded in each compiled
object's fatbinary, not loaded separately at process start. Archives link
statically (`cargo:rustc-link-lib=static=<group>`) into the final binary;
there is no runtime kernel-loading step.

`CHUNK_SIZE = 32` is defined once in Rust (`kv_cache/chunked/types.rs`) and
once in CUDA (`arena_table.cuh`, mirrored as `candle-kernels::CHUNK_SIZE`) —
the build treats a mismatch as a correctness bug, not a tuning knob.

| Directory | Contents |
|---|---|
| `paged-decode/` | Single-token-per-sequence attention over the paged KV cache; `SlotHeader`/`slot_types.cuh` absolute-pointer addressing |
| `paged-prefill/` | Multi-token prefill attention, INT8 context path |
| `paged-glue/` | Boundary gap-fill attention for reprojected context stitching |
| `quantized/` | Native quantized matmul (GEMV/GEMM ladder, greedy decomposition), `impl/` and `loader/` subtrees |
| `provenance/` | The BDP scan kernels — `bdp_scan.cu` (scalar), `bdp_bmma.cu` (1-bit tensor-core), `bdp_imma.cu` (INT8 tensor-core), shared `bdp_vote.cuh` |
| `sampling/` | The fused all-modifier sampling kernel |
| `simple/` | Small utility kernels, including `kv_migrate.cu` (the tier-migration copy kernel) |
| `dequant/`, `quantize/`, `convert/`, `mma/` | Format conversion and MMA building blocks shared across the above |

**Adding a kernel**: add the `.cu`/`.cuh` under the right `src/<subdir>/`,
register it in that subdir's archive group in `build.rs` (SHA-256 picks up
the change automatically), declare the `extern "C"` FFI signature in that
subdir's `api.rs` (re-exported from `candle-kernels/src/lib.rs`), and call it
`unsafe` from `candle-core/src/cuda_backend/`, or directly from
`candle-nn`/`candle-transformers` for paged/provenance kernels that touch
chunked-cache internals.

---

## 8. Conversation substrate & persistence

### 8.1 Substrate model

`substrate.rs`'s `Substrate` is the concrete per-session store of turns and
sections; `projection/` is a **pure structural reconciler** that compresses
that unbounded substrate into a fixed-size context window every
reprojection — it owns no content, no tokenizer, no scoring, only the
declared schema and the budget/selection rules. The schema (authored as
YAML, `projection/yaml.rs`, `projection/schema.rs`) declares a hierarchy:

```
Schema
├── SystemPrompt → [Section, Section, ...]        (static, shared by every layer)
└── Layers                                        (ordered; "dialogue", "repo_map",
    ├── Layer { score_formula, budget, ... }        "code_reading", etc.)
    │   └── Groups
    │       └── Group { selection rule, budget }
    │           └── Turns                          (append-only, opaque to this crate)
```

Selection rules are a closed set — `always_visible`, `top_k(k)`, `single`,
`conversation { recent, historical_top_k }` — evaluated by score, but turns
always **emit in insertion order** (score decides who's in-window; sequence
order decides how it reads). Design:
[`docs/conversation_builder.md`](docs/conversation_builder.md).

### 8.2 Timelines, turns, sections, collections

A **turn** is one user/assistant exchange (or one step of a repo_map folder
chain, or a code_read file scope — every ingest pipeline produces turns onto
some layer's group). A **timeline** (`TimelineId`) is a distinct KV-bearing
projection lineage — a re-scan or fork mints a new timeline rather than
mutating an old one, keeping provenance normalization scopes (§5.4) and KV
sharing well-defined. A **section** is a system-prompt fragment; a
**collection** is a named group of sections filled from a folder or the tool
registry — `zend/src/ingest.rs` derives load plans purely from the schema's
declared structure, so nothing about *how* to populate a layer lives in the
YAML itself.

### 8.3 The redo log and why persistence is mandatory

`candle-conversation/src/persistence/` is a generalized (not
conversation-specific) module: an append-only, content-addressed redo log at
`.substrate/substrate.log`, split into ~4 GiB segment files (`segment.rs`,
`segmented_log.rs` — [`docs/segmented_substrate_log.md`](docs/segmented_substrate_log.md))
once large. **There is no in-memory-only mode** — `Substrate` cannot be
constructed without a backing log; every turn append and section ingest goes
through this layer, no `Option<Persistence>` seam. One concern per file:
framing (`record.rs`), the append/fsync log itself (`log_file.rs`), the
skip-load header walk (`walker.rs`), the in-RAM last-writer-wins index
(`manifest.rs`), the recovery chains (`content_hash.rs`, `header_index.rs`,
`recovery.rs`), O(1) dead-byte accounting and whole-file rewrite
(`accounting.rs`, `compaction.rs`), multi-log/shared-base loading
(`inherit.rs`), the tier-migration orchestrator that drives `candle-nn`'s
`kv_pack`/`kv_unpack` (`transfer.rs`), the background thread (`thread.rs`),
warm→hot promotion (`elevate.rs`), and the NVMe→VRAM bridge
(`cold_load.rs`). Full design:
[`docs/kv_tier_migration.md`](docs/kv_tier_migration.md) §13.

---

## 9. The `zend` daemon (`zend/`)

`zend` is the binary that turns the conversation engine into a running
service: an OpenAI-compatible HTTP API (`zend/src/api/`, `axum` router in
`api/mod.rs` — `/v1/chat/completions`, `/v1/models`, `/v1/substrate/*`
introspection routes, `/v1/conversations/*`, a `/ws/logs` websocket, plus the
embedded web UI) fronting a `ZendSession` (`session.rs`) that owns the
`ConversationEngine`.

**Ingest** populates non-dialogue layers of the projection schema with no
per-layer code: `ingest.rs` derives a load plan purely from the schema's
declared structure (the `Sequence`-selection group is the live dialogue
layer and is excluded; `repo_map` and `code_reading` are recognised by
convention; everything else reads ChatML records from a same-named folder).
Two pipelines matter architecturally: **`repo_scan/`** walks the workspace and
mints one conversation per directory onto the `repo_map` layer, each explored as
two `code_read`-shaped tool round-trips — list the folder, then read its
`README`/module-doc anchor (`anchor.rs`) — the last of which **decodes** a
two-sentence summary of what the folder is for. Keeping every decode's request
one turn back is what holds it on task; folded into one longer chain the model
keeps driving the tool loop instead of answering. Both tool responses are produced by running the real tools, so a
prefilled response cannot drift from what the model sees at runtime. The unit's
content hash (`DirState`) covers the evidence the turns actually *show* — the
listed page plus the anchor excerpt — so a refresh re-ingests exactly the
directories whose shown evidence moved. **`code_read/`** mints one
conversation per file, parsed (`carve.rs`, `parsers/`) into scope-aware
parts each contributing a prefilled `read_file` tool round-trip; files
ingest **in parallel** (`CODE_READ_PARALLELISM` workers) while scopes within
one file ingest **serially** (a response turn's summary must decode with its
call turn in its projected prefix), so the parallelism is across files —
each worker's scope decodes coalesce into the scheduler's shared multi-session
batching, amortising the MoE expert load exactly as any other wave.

**Tools** (`tools.rs`, `tool_def.rs`, YAML specs under `prompts/tools/`) are
declared data — network sessions (SSH/TCP/TLS/HTTP/SQL), crypto primitives,
file/credential management, code execution — surfaced as the `tools`
section collection and selected into context by the same
provenance/normalization machinery as everything else (§5.4).

**Projection schema**: `zend/src/prompts/projection.yaml` is the concrete
schema zend runs — the layer list (`dialogue`, `repo_map`, `code_reading`,
identity/mood collections, …), each layer's `decode_priority` for the
continuous-fair-waves throttle (§6), and their budgets.

---

## 10. Concurrency and the VRAM governor

The scheduler (`candle-conversation/src/scheduler/`) is a **single background
thread** that owns all GPU state — model weights, KV arenas, the expert pool,
the provenance gallery arena. Caller threads never touch the GPU directly;
they submit `SchedulerRequest`s over a channel and receive `TurnHandle`s.

**Waves.** A wave is the unit of scheduling: a mixed forward whose rows are
the union of in-flight decode (`q_len == 1`), prefill, and glue tokens for
one step, built as one packed batch header spanning all three
(`docs/unified_wave_inference_engine.md`). Attention runs per-type (three
kernel launches, each into its disjoint row range of one output buffer);
everything after attention — `o_proj`, FFN norm, the MoE grouped GEMM — runs
**once** over all rows, which is where the PCIe-amortisation gain comes
from. Continuous fair waves (§6) decouple the decode cursor's per-wave full
sweep from prefill/glue's throttled creep so a large ingest never evicts the
live decode session's hot experts.

**Admission and the VRAM governor.** `candle-core/src/vram/` owns a
process-global `VramGovernor` per GPU: a "balloon and measure" boot step
establishes the true resident capacity `C` (not raw `cuMemGetInfo` total);
allocations are tagged by `AllocClass` (`Weights`, `Expert`, `Scratch`,
`Kv`) purely for accounting/forecasting, never as an availability gate —
`measure()` (DXGI on Windows, `cuMemGetInfo` on Linux) is the sole source of
truth. The KV budget is **observed, not predicted**: whatever headroom
remains once weights and experts are resident *is* the KV variable band,
floored by `kv_floor = KV_FLOOR_ABS + KV_FLOOR_PCT × (C − Weights)`. A
five-rung **relief ladder** (`Trivial → Cheap → Moderate → Costly →
Critical`) runs low-to-high as headroom tightens; only the top rung takes a
GPU sync. KV eviction to the warm tier registers at `Costly`; the
provenance gallery arena (§5.2) registers at the *cheapest* rungs so it is
reclaimed before KV is ever touched. Design:
[`docs/vram_governor_design.md`](docs/vram_governor_design.md).

---

## 11. Where to look

| Concern | Directory / file |
|---|---|
| HTTP API, daemon entry point | `zend/src/api/`, `zend/src/main.rs` |
| Repo/code ingest | `zend/src/repo_scan/`, `zend/src/code_read/` |
| Tool catalog | `zend/src/tools.rs`, `zend/src/prompts/tools/` |
| Conversation engine entry point | `candle-conversation/src/engine.rs` |
| Scheduler / wave loop | `candle-conversation/src/scheduler/run.rs`, `mod.rs` |
| Projection schema + builder | `candle-conversation/src/projection/` |
| Substrate (turns/sections storage) | `candle-conversation/src/substrate.rs` |
| Summary forest | `candle-conversation/src/summary_tree/`, `tree/` |
| Provenance capture + BDP scan (host) | `candle-conversation/src/provenance/` |
| Provenance gallery VRAM arena | `candle-conversation/src/provenance/gallery_arena/` |
| Score normalization | `candle-conversation/src/normalization/` |
| Redo log / tier migration (host) | `candle-conversation/src/persistence/` |
| Chunked KV cache (paging, arenas, GIDs) | `candle-nn/src/kv_cache/chunked/` |
| Adaptive quantization policy | `candle-nn/src/kv_cache/chunked/compression_policy.rs`, `sampled_selection/` |
| Migration kernel binding | `candle-nn/src/kv_cache/chunked/migrate.rs` |
| Batched inference API | `candle-transformers/src/models/batched_inference.rs`, `batched_model.rs`, `batched_layer.rs` |
| MoE model + expert pipeline | `candle-transformers/src/models/quantized_qwen3_moe.rs`, `expert_lre/` |
| VRAM governor | `candle-core/src/vram/` |
| CUDA kernels (AOT) | `candle-kernels/src/` (see §7 table), `build.rs`/`build_utils.rs` |
| Design docs (authoritative) | `docs/*.md` |
