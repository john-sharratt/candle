# docs/

Design documents for the unbounded-context inference engine, its two products
(Zen Code / `zend`, Battle Cities), and the research programme around them.

**These documents are authoritative.** Per the repo-wide rule in `AGENTS.md`:
when a design doc and the code disagree, the doc wins and the code is the bug
— fix the code (or, if the doc's design genuinely changed, fix the doc in the
same commit). Several documents below carry an explicit `Status:` line stating
how far implementation has progressed against the design (draft / built /
built-with-divergences); read that line before assuming the doc describes
exactly what ships today.

Every `.md` directly in `docs/` and in `docs/deepseek/` is listed below.
Superseded and historical material lives in `docs/archived/` — see
[`docs/archived/README.md`](archived/README.md). Do not treat anything there
as current design; where an archived document is still the only full account of
something, the row that points at it says so.

## Core architecture & theory

The foundational papers. Read these first for orientation; nearly everything
else in this directory extends one of them.

| Doc | Covers |
|---|---|
| [`unbounded_agents.md`](unbounded_agents.md) | **"One Card, One Stack", v2 (in preparation)** — the canonical technical report. The published **v1** is [`unbounded_agents_v1.md`](unbounded_agents_v1.md), archived unchanged with an erratum; *that* file is what `tokera.com/papers/one-card` renders live (see [`web_gateway_design.md`](web_gateway_design.md) §4), so an edit there is an edit to the public paper. Asymptotic Numerical Stability theorem (O(1) error at unbounded context depth), the four integrated system contributions (Markov expert prediction, adaptive KV quantization, attentional provenance indexing, three-tier paged context), and the measured throughput numbers. Read this first. |
| [`theory_of_the_mind.md`](theory_of_the_mind.md) | "The Conversations a Mind Needs" — functional decomposition of the cognitive substrate into parallel sub-conversations (specialists, affect, goals, relationships, self), argued from engineering constraints and clinical phenomenology. Underlies `npc_mind_design.md`, `cognitive_coding.md`, and `sdlc_agent.md`. |
| [`what_is_consciousness.md`](what_is_consciousness.md) | Substrate Composition Theory — consciousness as the compound effect of attention-weighted composition over specialised parallel sub-minds; reports an experimental protocol run on the substrate architecture. Read for the theoretical grounding behind the multi-layer substrate design, not for build details. |
| [`democratic-ai.md`](democratic-ai.md) | "Toward Democratic AI" — the position piece on what running the whole stack on one card, owned by its user, changes about who holds AI. |
| [`palquant.md`](palquant.md) | PalQuant — calibration-free 7× KV cache compression via adaptive per-block quantization; the paper-form account of the C0–C10 scheme. |

## Provenance & retrieval

Q-vector fingerprinting, the Binary Directional Provenance (BDP) scan, and
score selection — the mechanism that picks *what* enters context at each step.

| Doc | Covers |
|---|---|
| [`attention_provenance.md`](attention_provenance.md) | Attentional Provenance Indexing — the original paper on capturing live Q vectors as cognitive-state fingerprints, the three-depth-band fingerprint format, and the flat CPU scan that replaces hierarchical retrieval. |
| [`gallery_residency.md`](gallery_residency.md) | Which turns may be resident in the VRAM-resident gallery arena (`candle-conversation/src/provenance/gallery_arena/`), and when each kind goes: shared turns and conversation turns. How a page is laid out, uploaded and scanned is in the archived original, [`archived/paged_gallery_arena.md`](archived/paged_gallery_arena.md) (built, target met), including the tensor-core (`bdp_bmma.cu`) and scalar (`bdp_scan.cu`) scan backends. |
| [`gallery_dedup.md`](gallery_dedup.md) | The dedup overlay: near-duplicate tokens within a turn are kept off the card while the substrate keeps every token. |
| [`incremental_chunk_exclusion.md`](incremental_chunk_exclusion.md) | The anomalous-chunk exclusion learned incrementally from a fixed sample of calibration questions, kept as mergeable per-question records replicated in the substrate. |
| [`provenance_scoped_signatures.md`](provenance_scoped_signatures.md) | **Partly built.** Gallery windows cut by the same boundary formula and to the same shape as projection's probes: the boundary rule (`provenance::boundaries`), the windows (`provenance::probe_windows`) and the aggregations (`provenance::window_score`) are built and measured offline; the doc carries the gates — boundary parity, capture coverage, self-retrieval — and what remains. |
| [`provenance_score_normalization.md`](provenance_score_normalization.md) | **Built** (`candle-conversation/src/normalization/`). Normalizing raw per-candidate provenance scores against the candidate's background so they are comparable across candidates of very different scale. |
| [`provenance_adaptive_projection.md`](provenance_adaptive_projection.md) | Design for scoring coverage, attention budgets, turn locality, anchors, probe composition and consensus fusion in the projection policy. |
| [`hit_driven_projection.md`](hit_driven_projection.md) | A dialogue's slot holds what the model's own indexer keeps asking for, with a pool up to four times the slot standing behind it in the warm tier; what the indexer would have read is loaded at the next projection. |
| [`qsa_stratified_selection.md`](qsa_stratified_selection.md) | Stratified QSA selection on Qwen3.8-Flash-Next: the cells selected per query by indexer score, the pinned ranges, and the kernel and live recall. |
| [`qsa_index_q8.md`](qsa_index_q8.md) | Design for QSA index keys stored in Q8_0. |
| [`tool_provenance_distillation.md`](tool_provenance_distillation.md) | How a calibration turn's tokens/KV are discarded after its Q-signature (`WideQSig`) is captured, via a `Distilled` substrate marker — shrinks the tool-calibration corpus to signature-only. |
| [`tool_calibration.md`](tool_calibration.md) | How tools are calibrated for provenance selection. |
| [`tool_selection_provenance_results.md`](tool_selection_provenance_results.md) | **Shipped 2026-07-06.** The production tool-selection mechanism (per-token `WideQSig` folding, late-fusion `z × margin` scorer) and the `belief-*` harness subcommands used to evaluate it. §§1–22 are research history; §23+ is the shipped design. |

## KV cache, tiering & VRAM residency

The three-tier paged KV cache (GPU hot / RAM warm / NVMe cold), its
persistence layer, and the memory-residency design above it. The original
tier-migration design (`kv_pack`/`kv_unpack`, hot→warm→cold pipeline, redo-log
persistence; built with two known divergences, listed in `AGENTS.md`) is
archived at [`archived/kv_tier_migration.md`](archived/kv_tier_migration.md);
the documents below are the current ones.

| Doc | Covers |
|---|---|
| [`warm_lift.md`](warm_lift.md) | **Built.** The warm leg of `elevate_to_hot`: RAM → VRAM, batched and sync-free (`candle-nn/src/kv_cache/chunked/warm_lift/`, `lift_warm_to_gpu`), replacing the per-layer path. |
| [`kv_quantize_in_the_gap.md`](kv_quantize_in_the_gap.md) | **Design, not built.** K/V is compressed on the scheduler thread in the gap where it is made, and persistence only copies. |
| [`kv_drain_under_pressure.md`](kv_drain_under_pressure.md) | A quant reserve, and compaction that never blocks migration; superseded in part by `kv_quantize_in_the_gap.md`. |
| [`held_ground_elimination.md`](held_ground_elimination.md) | **Built 2026-10-18.** How held ground was eliminated from the partition, with what was measured. |
| [`vram_governor_design.md`](vram_governor_design.md) | Cross-platform module measuring real free VRAM: the startup half (probes, balloon-and-measure, `AllocClass` tags, the capacity arithmetic the span is sized from) is built; the runtime criticality ladder was removed when KV moved onto a static reservation. |
| [`vram_span_partition.md`](vram_span_partition.md) | **Current.** How device memory is laid out and handed out today: one span shared by persist staging, KV regions, the per-forward transient tier and the expert weight zone, with the elastic `weight_floor` between the last two. Written for a newcomer; covers the flow of weights, transients and experts, the placement rules and why each is a refusal, the measured cost of the KV side not compacting, and the continuous left-side compaction that fixes it. |
| [`driver_model.md`](driver_model.md) | The driver model — WDDM, TCC, Linux — read from the device, and how it changes the memory arithmetic. |
| [`archived/elastic_vram_partition.md`](archived/elastic_vram_partition.md) | **Historical design argument** for the span partition. Replaced the static `kv_floor` / `expert_budget` split with one reservation and a moving boundary: KV arenas grow left-to-right, expert slots right-to-left, the boundary renegotiated at the expert pipeline's end-of-pass. §14 records where the build diverged from the design and why. |

## MoE & expert cache

Expert prefetch/eviction prediction and the expert cache that streams a model
larger than the card.

| Doc | Covers |
|---|---|
| [`expert_cache_design.md`](expert_cache_design.md) | **Built, and measured 12–87 % faster than the two-tier cache it replaces** (every config of `test_parallel_batched_forwarding`). VRAM ↔ RAM ↔ NVMe over an authoritative repacked pack file: with a cold tier that always holds a copy, eviction is a drop, the elastic partition's boundary is uncapped, and a restart skips the 42 s repack. §12 records what the build settled. |
| [`moe_live_dispatch_design.md`](moe_live_dispatch_design.md) | **Implemented (revision 2, §0).** The expert forward for every `ExpertCache` model with no host round trip: `bucketize` classifies routed experts VRAM / pinned / cold against a live pointer table, the expert kernels' worker blocks copy pinned misses into VRAM scratch and compute them, a stager thread (no CUDA calls) stages cold experts from NVMe, and the routing summary is read asynchronously from a mapped ring. Supersedes [`archived/gpu_native_moe_dispatch.md`](archived/gpu_native_moe_dispatch.md). |
| [`self_contained_model_packs.md`](self_contained_model_packs.md) | **Built (2026-10-10).** The repacked pack is the model's only run-time file — every tensor, the metadata and the tokenizer in one GGUF container (`candle-transformers/src/models/model_pack/`). The repack fingerprint is per format, so a repack change invalidates only the packs that contain that format. Supersedes `expert_cache_design.md` §5.2. |
| [`elastic_warm_tier.md`](elastic_warm_tier.md) | **Built (2026-11-21).** The elastic warm tier — pageable expert RAM that follows the machine. |
| [`cpu_expert_compute.md`](cpu_expert_compute.md) | **Built (§9 steps 1–5).** CPU expert compute: the host pool, its rings, the CPU class in `moe_bucketize`, and the partition's refill price. |
| [`archived/markov_expert_prediction_eval.md`](archived/markov_expert_prediction_eval.md) | **Archived evaluation.** "The Markov Wave" — ID-only Markov models for MoE expert promotion/eviction prediction, evaluated by 21-fold cross-validation on a captured Qwen3-30B-A3B routing trace. The live predictor is in `expert_lre/transition.rs` and `expert_lre/blend.rs`. |

## Scheduler & wave engine

The wave scheduler that steps many concurrent sessions through decode, prefill
and glue together, and the machinery around one wave.

| Doc | Covers |
|---|---|
| [`continuous_fair_waves.md`](continuous_fair_waves.md) | **Supersedes `archived/unified_wave_inference_engine.md`.** Runs decode and prefill concurrently at decoupled per-layer rates instead of time-sharing a flip, so decode's hot-expert working set is never smashed by a background prefill; the `decode_priority` bands (`Low`/`Normal`/`High`/`Urgent`), airtime debt, time-sliced quanta and mid-wave admission. |
| [`async_wave_submission.md`](async_wave_submission.md) | **Built.** How work reaches the wave engine without a thread blocked on every in-flight turn: `flume` channels, awaitable turns, and npcd rebuilt as supervised character tasks so one slow mind never stalls the cast. |
| [`wave_feeder.md`](wave_feeder.md) | The wave feeder — a fat pipe, a smart valve, and a simulator to tune it. §4.11 is the admission design as built; §4.1–§4.10 are the kept record of what was tried before it and why each attempt was falsified. |
| [`wave_step.md`](wave_step.md) | The wave step: where a residual lives between forwards (the co-batched step and the creep), for how long, and what it costs. |
| [`prefill_gangs.md`](prefill_gangs.md) | How a prefill group divides its budget between the prompts waiting on it, and why background ingest is divided differently from dialogue. |
| [`decode_graphs.md`](decode_graphs.md) | Decode CUDA graphs — the hybrid wave run as multiple graphs chained with no host sync (`candle-core` `cuda_backend/graph/`), with status per model and the measurements that chose the design. |
| [`phased_buffers.md`](phased_buffers.md) | The transient tier placed by a typed program: how the device ground one forward's intermediates live on is bought and placed. |
| [`cooperative_admission.md`](cooperative_admission.md) | How a zend cluster places turns across members: decodes concentrated where weights are resident, prefill kept off the decode set's card. Placement itself is now [`static_placement.md`](static_placement.md). |
| [`static_placement.md`](static_placement.md) | Where every unit of work in the cluster runs, decided from a table the deployment states rather than from figures the members measure; replaces the measured placement of `cooperative_admission.md`. |

## Recurrent (DeltaNet) models

The hybrid Qwen3.5/3.6/3.8 lineage, whose recurrent state has to be saved,
forked and rewound like KV.

| Doc | Covers |
|---|---|
| [`deltanet_state_persistence.md`](deltanet_state_persistence.md) | Switching to the hybrid: recurrent state, glue and provenance — how the state is sealed, restored and forked across turns. |
| [`deltanet_timeline.md`](deltanet_timeline.md) | **Implemented.** One state buffer per layer with a timeline of rows, and a partial accept that picks a row (replacing the verify stash). |
| [`recurrent_state_behavioural_tests.md`](recurrent_state_behavioural_tests.md) | The behavioural test catalogue for conversational memory on a recurrent stack. |
| [`progressive_yarn.md`](progressive_yarn.md) | **Built.** A per-slot RoPE schedule for every model. |

## DeepSeek-V4-Flash (`docs/deepseek/`)

The native-sparse 1M-context port of the sparse-latent MoE family
(`candle-transformers/src/models/latent_moe/`).

| Doc | Covers |
|---|---|
| [`deepseek/deepseek_v4_flash.md`](deepseek/deepseek_v4_flash.md) | DeepSeek-V4-Flash support — the design document. |
| [`deepseek/deepseek_batched_paged_attention_plan.md`](deepseek/deepseek_batched_paged_attention_plan.md) | Batched/paged hybrid attention and kernelization plan: architecture foundations, implementation plan, `paged-deepseek` kernel design, and the test-driven migration plan. |
| [`deepseek/deepseek_hot_path_invariants.md`](deepseek/deepseek_hot_path_invariants.md) | The study behind the hot-path invariants in `AGENTS.md` (the compute-bound prefill and decode targets and every violation found, with references). Authoritative. |
| [`deepseek/deepseek_int8_hot_corpus.md`](deepseek/deepseek_int8_hot_corpus.md) | The int8-hot corpus and BF16 rope bands for the latent attention. |
| [`deepseek/deepseek_prefill_tensorcore_pv.md`](deepseek/deepseek_prefill_tensorcore_pv.md) | The latent prefill kernel (HEAD_DIM=512): tensor-core PV, 2-block occupancy, pre-quantized V corpus. |
| [`deepseek/deepseek_turn_seal_persistence.md`](deepseek/deepseek_turn_seal_persistence.md) | Turn seal and resume: window-ring snapshot, closed corpus and Indexer wide-Q. |
| [`deepseek/deepseek_v4_speculative_decode.md`](deepseek/deepseek_v4_speculative_decode.md) | DSpark speculative decode for DeepSeek-V4-Flash (in progress). |

## Conversation substrate & projection

How the unbounded substrate compresses into a fixed context window every turn.
The original projection-builder design
([`archived/conversation_builder.md`](archived/conversation_builder.md)) and the
immutable summary forest
([`archived/immutable_summary_forest.md`](archived/immutable_summary_forest.md))
are archived; the code under `candle-conversation/src/projection/` and
`summary_tree/` is the current form.

| Doc | Covers |
|---|---|
| [`stencil_tree.md`](stencil_tree.md) | **Implemented.** Schema-guided constrained decoding (`candle-conversation/src/stencil/`) — forces tool calls to an exact catalog name + JSON schema, plus a fourth "think steering" front-end for `<think>` block control. |
| [`submitted_sections.md`](submitted_sections.md) | Per-conversation collection members: sections a conversation submits for itself, how they are sealed and addressed. |
| [`request_steps.md`](request_steps.md) | **Built (2026-11-19).** A request's steps, and the request turn that anchors them. |
| [`thinking_span_projection.md`](thinking_span_projection.md) | **Implemented, authoritative.** How a turn's thinking span is projected. |
| [`projection_swap_cap.md`](projection_swap_cap.md) | A mid-decode reprojection brings only a few new conversations into each selected group; the rest wait for the next reprojection. |
| [`substrate_lock_free.md`](substrate_lock_free.md) | The substrate without locks: the audit of every critical section around a substrate, its redo log and zend's engine, and the design that removed them. |
| [`guidance_layer.md`](guidance_layer.md) | Guides and journals placed by provenance, calibrated as they are used. |
| [`shared_journal_sections.md`](shared_journal_sections.md) | Shared journal sections. |
| [`turn_handover.md`](turn_handover.md) | **Design (2026-11-18).** A member that leaves hands its turns on. |

## The zend product & tool system

Zen Code: the persistent coding-assistant daemon, its tool surface, and its
web UI.

| Doc | Covers |
|---|---|
| [`coding_assistant.md`](coding_assistant.md) | Zen Code top-level design: the `zend` daemon + `zen-vscode` (Continue fork) + read-only web chat, phased-ingestion trunk sessions, and how the inference engine's primitives (provenance indexing, KV tiering, quantization) are put to work for a coding assistant. |
| [`cognitive_coding.md`](cognitive_coding.md) | Extends the inference stack into a coding-agent architecture that continuously pulls context during decode (driven by attention) instead of one-shot RAG retrieval before generation. |
| [`sdlc_agent.md`](sdlc_agent.md) | "The Vertically-Complete Engineering Agent" — extends Zen Code's substrate vertically across 28 layers spanning silicon-to-operations, so the agent reasons across the whole software lifecycle in one mind rather than one layer. |
| [`paper_conversation_replay.md`](paper_conversation_replay.md) | **The acceptance test for scoring, provenance and working-set work.** Four messages replayed on a fresh conversation (tour → the paper → improve it → make the changes): how to run and grade it, what a good projection holds per turn, the 2026-10-25 baseline, and a verified answer key. |
| [`tool-system.md`](tool-system.md) | Specifies the server-registered tools, the client-specific tool semantics (Continue client-executed vs. web-chat server-executed), and the Hermes-style `<tools>`/`<tool_call>` prompt format. |
| [`web_search_design.md`](web_search_design.md) | The `web_*` tool family (`web_search`, `web_fetch`, `web_extract`, `web_compare`, `web_deep_research`) built on the Gemini API's combined grounded-search + Structured-Outputs capability. |
| [`zend_working_set.md`](zend_working_set.md) | **Built.** The working set — what a dialogue already knows about the code. |
| [`zend_layers.md`](zend_layers.md) | How layers above `code_reading` are built; zend's schema declares none now, and the engine is kept for per-file and per-day units. |
| [`zend_branch_ingest.md`](zend_branch_ingest.md) | **Built and unit-tested.** Ingesting repository branches as origin holds them. |
| [`zend_git.md`](zend_git.md) | The git layer of the `zend-vfs` crate and the model-facing `git_*` tools; the VFS purge is designed, not built. |
| [`zend_journal.md`](zend_journal.md) | **Built.** zend's journal, adapted from npcd's. |
| [`zend_vfs_events.md`](zend_vfs_events.md) | **Built.** A conversation's files as substrate events. |
| [`zend_workspace_execution.md`](zend_workspace_execution.md) | Workspace, tooling and execution architecture: the workspace and tools are built, the single-machine sandbox runs as `run_command`, cluster execution is designed, not built. |
| [`zend_ui_redesign.md`](zend_ui_redesign.md) | **Built.** Phased plan for the Zend web UI: projection timeline gutter, windowed-substrate inspector, per-part upload progress, thinking-effort/answer-length composer dials, theme switcher. |
| [`zen_chat.md`](zen_chat.md) | Zen chat: a private conversation, deleted when it is done. |
| [`zen_everywhere.md`](zen_everywhere.md) | **Design, agreed 2026-11-15.** Zen everywhere, drones at rest. |
| [`zen_tool_reach.md`](zen_tool_reach.md) | **Built.** Closing the gap between Zen's tools and Claude's. |
| [`guest_access.md`](guest_access.md) | Guests: an allowlist, a role below User, and a queue. |
| [`guest_draw_orchestration.md`](guest_draw_orchestration.md) | One call that draws now, authorizes meanwhile, and releases on both. |
| [`build_agent.md`](build_agent.md) | Builds that a planned restart never cuts short. |

## MCP, memory, journals & jobs

The connector Claude uses, the memory behind it, and the jobs it watches.

| Doc | Covers |
|---|---|
| [`claude_zend_gateway.md`](claude_zend_gateway.md) | Claude and zend: one orchestrator, one workhorse. Status line: designed, not built; authoritative over the remote connector. |
| [`mcp_app.md`](mcp_app.md) | MCP as an aggregator over the apps' own servers. |
| [`mcp_guides.md`](mcp_guides.md) | **Built (2026-10-26).** A skill library the connector serves. |
| [`mcp_user_context.md`](mcp_user_context.md) | **Built (2026-10-29).** Memory and a per-client guide over MCP. |
| [`mcp_file_tools.md`](mcp_file_tools.md) | **Design, not built.** File, log and site tools in the Zen MCP, and the gateway's pub/sub. |
| [`jobs.md`](jobs.md) | **Built (2026-11-19).** Everything that runs longer than a call is a job, watched, read, searched, listed and stopped with the same five tools. |
| [`journal.md`](journal.md) | The journal: a short, dated record a character keeps beyond the tail of its conversation. |
| [`memory_backbone.md`](memory_backbone.md) | **Redesigned, for review.** One master journal per group, in git. |
| [`zen_memory_and_guides.md`](zen_memory_and_guides.md) | Zen's memory, guides and task driving: the build plan from the 2026-11-08 audits; the branch is merged to `main`. |
| [`private_mode.md`](private_mode.md) | **Building.** Private mode and diaries. |
| [`cluster_status_story.md`](cluster_status_story.md) | **Partly built.** A failed install: what the member does, and what `cluster_status` says. |
| [`memory_census.md`](memory_census.md) | The memory census: counting what motherd allocates, what the OS has committed for it, and what each substrate holds. |

## Battle Cities: NPCs & the narrative engine

| Doc | Covers |
|---|---|
| [`npc_mind_design.md`](npc_mind_design.md) | "The Asynchronous Mind" — the NPC cognitive architecture: one mechanism (gather substrate under salience, attend, act, write back) applied uniformly, with a mutable substrate the model influences but never controls and an immutable core it cannot write. |
| [`asynchronous_mind_hierarchy.md`](asynchronous_mind_hierarchy.md) | The asynchronous mind's hierarchy, invented tools, and learned memory. |
| [`reflection_and_dreams.md`](reflection_and_dreams.md) | The second and third conversations a character runs: `reflect` as a one-shot transient conversation over a deep dream gather, and a private per-NPC dream corpus maintained as a quality-diversity search — anti-correlated generation, relevance selection, recency reaping. Implements the reflection half of `theory_of_the_mind.md`'s layers 14/16/17/19. |
| [`narrative_engine.md`](narrative_engine.md) | Roleplay Engine (RPE) design: translates structured player `Input` events into narrated prose via a conversation-as-KV-cache narrator, with `parse_turn`/`text_to_inputs` as the crate's entry points. |
| [`npc_engine_design.md`](npc_engine_design.md) | The NPC engine, `npcd`. The HTTP surface and console it defers to the gateway doc for are described in [`web_gateway_design.md`](web_gateway_design.md). |
| [`npc_conversations.md`](npc_conversations.md) | **Built.** Conversations an NPC holds one-to-one beside its main loop, as the same person. |
| [`npcd_worlds_and_layers.md`](npcd_worlds_and_layers.md) | Worlds, layers and the shared corpus. |
| [`npcd_cast_lock_free.md`](npcd_cast_lock_free.md) | The cast without a lock. |
| [`npcd_anchored_edits.md`](npcd_anchored_edits.md) | **Implemented.** Anchored document edits in npcd and zend. |
| [`npcd_workflows.md`](npcd_workflows.md) | npcd operation workflows; [`npcd_workflows_current.yaml`](npcd_workflows_current.yaml) is the hardcoded pipeline as it runs, in the workflow format. |
| [`npcd_guardian.md`](npcd_guardian.md) | npcd guardian: proposal. |
| [`npcd_discord.md`](npcd_discord.md) | npcd on Discord. |
| [`mission_generator.md`](mission_generator.md) | The Command Table's mission generator. |
| [`mission_templates.md`](mission_templates.md) | Mission templates. |
| [`the_effector.md`](the_effector.md) | The effector device and `http://local`: the fixed surface, the dynamic world API, and the stencil that makes the model call it well. |
| [`tool_world_acts.md`](tool_world_acts.md) | Acting on the world — bodies, Battle Cities, and the Tower. |
| [`vault_world.md`](vault_world.md) | The Vault. |
| [`world_rename.md`](world_rename.md) | Renaming a world. |
| [`house_modes.md`](house_modes.md) | **Built (2026-11-09).** The house has one mode at a time. |
| [`zen_moon_mind.md`](zen_moon_mind.md) | Zen, the Moon's Mind. |
| [`battlecities_app.md`](battlecities_app.md) | Battle Cities on the cluster. |
| [`yenbot_app_host.md`](yenbot_app_host.md) | **Design, not built.** Yen Bots: an app host for Yen. |
| [`yenbot_preview.md`](yenbot_preview.md) | **Design, not built.** Yen Bots previews: an app on claude.ai, published from the page. |

## ZenJ: Zen as DJ

| Doc | Covers |
|---|---|
| [`zenj.md`](zenj.md) | **Built (`zenj/`, 2026-11-08); the spikes of §12 run on BlackWorld against it; first use New Year's Eve 2026.** Zen running the music on BlackWorld: rekordbox 7.2.19 as the engine with the DDJ-FLX4 attached, TIDAL for audio, Spotify for taste and phone requests; the `zenj` cluster member; the brain (set plan, next-track shortlist with Zen choosing, transition plans), the beat-accurate conductor over a virtual MIDI controller, human/assist/ZenJ hand-off, the four-screen visuals, failure ladder, spikes and build order. |

## Web tier & deployment

| Doc | Covers |
|---|---|
| [`web_gateway_design.md`](web_gateway_design.md) | **Built.** The `web` crate: one front door for the estate. Host→site→upstream routing with layered content roots, tokera.com rendered from markdown (blog and the papers, read live from this directory), OIDC sign-in whose parent-domain cookie carries one account across `code.` and `bot.`, per-upstream backoff with readable error pages, and websocket tunnelling. Read before touching anything under `web/`. |
| [`shared_model_host.md`](shared_model_host.md) | **Built (phases 1–7, §6.1).** One host process for every daemon: the `host` crate's startup, server, shared endpoints and identity rule; zend and npcd as its tenants on one engine core, each with its own substrate bound to the prescribed model; `motherd`, both in one process. Read before touching `host/`, a tenant's routes, or how a daemon starts. |
| [`replicated_substrate.md`](replicated_substrate.md) | **Built in §15's order (chunks 1–10).** zend, npcd and CPU agents started with `--cluster` dial one or more active-active gateways over one authenticated WebSocket per app and become their upstreams; every substrate record replicates to the other members, applied so arrival order never changes the result, with sender retransmit and a header-first catch-up; a member cut off keeps writing and rejoins behind a Paxos-decided generation; Paxos among the members decides world authorities and group values; file-backed state moves into VFS records published by git merges that agents run; a per-app build floor, model lock, A/B certification and self-installing rollouts govern builds. |
| [`substrate_git.md`](substrate_git.md) | Git in the substrate: repositories replicate as packs (stages 1–4 built). |
| [`deployment.md`](deployment.md) | Which machine runs what, the recorded launch lines `/up` replays, the gateway's TLS entrance, and the health checks. |

## Hardware & measured results

Workstation build specs and standalone quality/throughput measurements.

| Doc | Covers |
|---|---|
| [`performance.md`](performance.md) | Performance — the fleet: headline results at the top, every row per machine, and the published figures they are compared with. Quote numbers from here. |
| [`performance_series.md`](performance_series.md) | How the performance readings (`/v1/telemetry`'s engine half and `/v1/phases`) are answered, and how a cluster's are added up. |
| [`hardware_machine_consumer.md`](hardware_machine_consumer.md) | Parts list and cost for the AM5 + single RTX PRO 5000 Blackwell (72 GB) Battle Cities workstation build. |
| [`hardware_machine_enterprise.md`](hardware_machine_enterprise.md) | Parts list and cost for the 4× RTX PRO 4500 Blackwell Threadripper PRO production workstation build, sized to saturate aggregate GPU ingress via a 16-way PCIe 5.0 NVMe array. |

## Other files in this directory

`results/` holds the dated measurement records (per-machine baselines, width-gate
sweeps, the dependency battery runs, the provenance gates and scoring lab).
`provenance_expectations/` holds the expected working sets the provenance gates
and the dependency battery grade against. `images/` holds figures referenced by
the documents above.
