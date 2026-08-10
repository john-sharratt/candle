# docs/

Design documents for the unbounded-context inference engine, its two products
(Zen Code / `zend`, Battle Cities), and the research programme around them.

**These documents are authoritative.** Per the repo-wide rule in `CLAUDE.md`:
when a design doc and the code disagree, the doc wins and the code is the bug
— fix the code (or, if the doc's design genuinely changed, fix the doc in the
same commit). Several documents below carry an explicit `Status:` line stating
how far implementation has progressed against the design (draft / built /
built-with-divergences); read that line before assuming the doc describes
exactly what ships today.

Superseded and historical material lives in `docs/archived/` — see
[`docs/archived/README.md`](archived/README.md). Do not treat anything there
as current design.

## Core architecture & theory

The foundational papers. Read these first for orientation; nearly everything
else in this directory extends one of them.

| Doc | Covers |
|---|---|
| [`unbounded_agents.md`](unbounded_agents.md) | **"One Card, One Stack"** — the canonical technical report. Asymptotic Numerical Stability theorem (O(1) error at unbounded context depth), the four integrated system contributions (Markov expert prediction, adaptive KV quantization, attentional provenance indexing, three-tier paged context), and the measured throughput numbers. Read this first. |
| [`theory_of_the_mind.md`](theory_of_the_mind.md) | "The Conversations a Mind Needs" — functional decomposition of the cognitive substrate into parallel sub-conversations (specialists, affect, goals, relationships, self), argued from engineering constraints and clinical phenomenology. Underlies `npc_mind_design.md`, `cognitive_coding.md`, and `sdlc_agent.md`. |
| [`what_is_consciousness.md`](what_is_consciousness.md) | Substrate Composition Theory — consciousness as the compound effect of attention-weighted composition over specialised parallel sub-minds; reports an experimental protocol run on the substrate architecture. Read for the theoretical grounding behind the multi-layer substrate design, not for build details. |

## Provenance & retrieval

Q-vector fingerprinting, the Binary Directional Provenance (BDP) scan, and
score selection — the mechanism that picks *what* enters context at each step.

| Doc | Covers |
|---|---|
| [`attention_provenance.md`](attention_provenance.md) | Attentional Provenance Indexing — the original paper on capturing live Q vectors as cognitive-state fingerprints, the three-depth-band fingerprint format, and the flat CPU scan that replaces hierarchical retrieval. |
| [`paged_gallery_arena.md`](paged_gallery_arena.md) | **Built, target met.** The VRAM-resident paged arena that holds the provenance gallery for the belief scan, plus the tensor-core (`bdp_bmma.cu`) and scalar (`bdp_scan.cu`) scan backends. Read when touching the GPU-side provenance scan or its occupancy/kernel tuning. |
| [`provenance_score_normalization.md`](provenance_score_normalization.md) | Design (not yet built) for normalizing raw per-candidate provenance scores so they're comparable across candidates of very different scale (e.g. workspace-root vs. a single subdirectory). |
| [`tool_provenance_distillation.md`](tool_provenance_distillation.md) | How a calibration turn's tokens/KV are discarded after its Q-signature (`WideQSig`) is captured, via a `Distilled` substrate marker — shrinks the tool-calibration corpus to signature-only. |
| [`tool_selection_provenance_results.md`](tool_selection_provenance_results.md) | **Shipped 2026-07-06.** The production tool-selection mechanism (per-token `WideQSig` folding, late-fusion `z × margin` scorer) and the `belief-*` harness subcommands used to evaluate it. §§1–22 are research history; §23+ is the shipped design. |

## KV cache, tiering & VRAM residency

The three-tier paged KV cache (GPU hot / RAM warm / NVMe cold), its
persistence layer, and the memory-residency governor above it.

| Doc | Covers |
|---|---|
| [`kv_tier_migration.md`](kv_tier_migration.md) | **Design v7, built with two known divergences.** The full VRAM↔RAM↔NVMe migration path: `kv_pack`/`kv_unpack` kernels, hot→warm→cold pipeline, redo-log persistence. The canonical reference for the tiering system; states exactly what shipped vs. the target shape. |
| [`coresident_kv_metadata.md`](coresident_kv_metadata.md) | Moves per-chunk KV-head metadata (palette maps, formats, scales, device pointers) to a device-resident record living next to the KV bytes, removing a per-forward host reconstruct/PCIe-ship of that state. |
| [`glue_prefill_kernel.md`](glue_prefill_kernel.md) | **Implemented.** Batches the structural "glue" prefill (inter-turn boundary tokens re-derived every projection) into a single GPU pass via a `GAP_FILL`-specialized paged-prefill kernel. |
| [`segmented_substrate_log.md`](segmented_substrate_log.md) | **Implemented, rev 5.** The on-disk redo log design: ~4 GB segment files, per-segment maintenance (drop/compact/combine), and the removal of load-path compaction in favour of fully-background reclaim. |
| [`vram_governor_design.md`](vram_governor_design.md) | Cross-platform module owning GPU VRAM residency: measures real free VRAM, classifies allocations by evictability, and relieves pressure via a criticality-ranked callback ladder. Replaces ad hoc split-brain budgeting between the KV path and expert-slot sizing. |
| [`elastic_vram_partition.md`](elastic_vram_partition.md) | **Built.** Replaces the static `kv_floor` / `expert_budget` split with one reservation and a moving boundary: KV arenas grow left-to-right, expert slots right-to-left, and the boundary between them is renegotiated at the expert pipeline's end-of-pass. Supersedes `vram_governor_design.md` §7/§11 (both deleted). §14 records where the build diverged from the design and why. |

## MoE & scheduling

Expert prefetch/eviction prediction and the wave scheduler that steps many
concurrent sessions through decode/prefill/glue together.

| Doc | Covers |
|---|---|
| [`markov_expert_prediction_eval.md`](markov_expert_prediction_eval.md) | "The Markov Wave" — ID-only Markov models for MoE expert promotion/eviction prediction, evaluated by 21-fold cross-validation on a captured Qwen3-30B-A3B routing trace. Reports the promotion scoring rule, the LFRU eviction rule, and the wave-batched execution framing. |
| [`gpu_native_moe_dispatch.md`](gpu_native_moe_dispatch.md) | **Built and verified.** Eliminates the per-layer expert-routing GPU→CPU readback (the dominant WDDM decode stall) with fully GPU-native dispatch; measured before/after decode-forward and tokens/s. |
| [`expert_cache_design.md`](expert_cache_design.md) | **Design, not built.** Replaces the two-tier VRAM↔pinned-RAM expert cache with VRAM ↔ RAM ↔ NVMe over an authoritative repacked pack file. The current tiers are mutually *exclusive*, which forces eviction to be a 68 GB/config D2H copy of read-only weights and makes VRAM's floor a function of how much host RAM was allocated. With a cold tier that always holds a copy, eviction becomes a drop and the elastic partition's boundary is freed. Prerequisite for `elastic_vram_partition.md`, not an optimisation. |
| [`unified_wave_inference_engine.md`](unified_wave_inference_engine.md) | Design draft (pre-implementation, superseded by `continuous_fair_waves.md`). Packs decode+prefill+glue into one forward with a small/large batch flip and cooldown, to amortise all-expert PCIe streaming cost on VRAM-constrained hardware. |
| [`continuous_fair_waves.md`](continuous_fair_waves.md) | **Supersedes `unified_wave_inference_engine.md`.** Runs decode and prefill concurrently at decoupled per-layer rates instead of time-sharing a flip, so decode's hot-expert working set is never smashed by a background prefill. |
| [`wave_blocking_optimizations.md`](wave_blocking_optimizations.md) | Measured audit of scheduler-thread blocking and GPU sync dominating wave-latency variance, including a decode-corruption incident, its root cause, and the safe mitigation plan that avoids extra CUDA streams. |

## Conversation substrate & projection

How the unbounded substrate compresses into a fixed context window every turn.

| Doc | Covers |
|---|---|
| [`conversation_builder.md`](conversation_builder.md) | The `candle-conversation` crate's projection design: budget reconciliation across layers/groups/sections, and emission of a system prompt + ordered turn list at projection time. Explicitly scopes what the crate does *not* own (content storage, tokenization, scoring). |
| [`immutable_summary_forest.md`](immutable_summary_forest.md) | **Authoritative**, supersedes the AVL-tree summary structure in `docs/archived/infinite_conversations.md` §7. Append-only, immutable summary nodes whose parent is fixed by arrival order, eliminating the `dirty`-bit rebalancing machinery entirely. |
| [`stencil_tree.md`](stencil_tree.md) | **Implemented.** Schema-guided constrained decoding (`candle-conversation/src/stencil/`) — forces tool calls to an exact catalog name + JSON schema, plus a fourth "think steering" front-end for `<think>` block control. |

## The zend product & tool system

Zen Code: the persistent coding-assistant daemon, its tool surface, and its
web UI.

| Doc | Covers |
|---|---|
| [`coding_assistant.md`](coding_assistant.md) | Zen Code top-level design: the `zend` daemon + `zen-vscode` (Continue fork) + read-only web chat, phased-ingestion trunk sessions, and how the inference engine's primitives (provenance indexing, KV tiering, quantization) are put to work for a coding assistant. |
| [`cognitive_coding.md`](cognitive_coding.md) | Extends the inference stack into a coding-agent architecture that continuously pulls context during decode (driven by attention) instead of one-shot RAG retrieval before generation. |
| [`sdlc_agent.md`](sdlc_agent.md) | "The Vertically-Complete Engineering Agent" — extends Zen Code's substrate vertically across 28 layers spanning silicon-to-operations, so the agent reasons across the whole software lifecycle in one mind rather than one layer. |
| [`tool-system.md`](tool-system.md) | Specifies all 93 server-registered tools, the client-specific tool semantics (Continue client-executed vs. web-chat server-executed), and the Hermes-style `<tools>`/`<tool_call>` prompt format. |
| [`web_search_design.md`](web_search_design.md) | The `web_*` tool family (`web_search`, `web_fetch`, `web_extract`, `web_compare`, `web_deep_research`) built on the Gemini API's combined grounded-search + Structured-Outputs capability. |
| [`zend_ui_redesign.md`](zend_ui_redesign.md) | **Authoritative, edited in place.** Phased plan for the Zend web UI: projection timeline gutter, windowed-substrate inspector, per-part upload progress, thinking-effort/answer-length composer dials, theme switcher. |

## Battle Cities: NPCs & the narrative engine

| Doc | Covers |
|---|---|
| [`npc_mind_design.md`](npc_mind_design.md) | "The Asynchronous Mind" — the NPC cognitive architecture: one mechanism (gather substrate under salience, attend, act, write back) applied uniformly, with a mutable substrate the model influences but never controls and an immutable core it cannot write. |
| [`narrative_engine.md`](narrative_engine.md) | Roleplay Engine (RPE) design: translates structured player `Input` events into narrated prose via a conversation-as-KV-cache narrator, with `parse_turn`/`text_to_inputs` as the crate's entry points. |

## Hardware & measured results

Workstation build specs and standalone quality/throughput measurements.

| Doc | Covers |
|---|---|
| [`hardware_machine_consumer.md`](hardware_machine_consumer.md) | Parts list and cost for the AM5 + single RTX PRO 5000 Blackwell (72 GB) Battle Cities workstation build. |
| [`hardware_machine_enterprise.md`](hardware_machine_enterprise.md) | Parts list and cost for the 4× RTX PRO 4500 Blackwell Threadripper PRO production workstation build, sized to saturate aggregate GPU ingress via a 16-way PCIe 5.0 NVMe array. |
| [`perplexity_results.md`](perplexity_results.md) | WikiText-2 perplexity measurements across model/quant combinations (Qwen3-30B-A3B, Qwen2/3, Llama-3.2-3B), methodology and per-context-size result tables. |

## Other files in this directory

`calib_baseline_decode.tsv` is a raw calibration data file (not a design
doc). `images/` holds figures referenced by the documents above.
