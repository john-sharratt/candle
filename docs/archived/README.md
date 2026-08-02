# docs/archived/

Superseded and historical design documents. **Do not treat anything in this
directory as authoritative, and do not ingest it as current design.** Every
document here has been replaced by a later design (usually one in `docs/`
itself — see [`docs/README.md`](../README.md)), abandoned as a negative
result, or overtaken by a rewrite of the system it described. Several were
written before the current `unbounded_agents.md` / `kv_tier_migration.md`
architecture existed and describe mechanisms (AVL summary trees, two-tier
expert caching without Markov prediction, non-paged KV storage) that the live
code no longer has.

## Early whole-system papers (superseded)

| Doc | Covers |
|---|---|
| `attention_tree_paper.md` | Attention-organized conversation B-trees with regenerative KV warming — the pre-provenance-scan approach to unbounded dialogue history. |
| `high_density_inference_paper.md` | Trie-constrained generation as a retrieval mechanism, plus dynamic-fact override semantics — an earlier retrieval architecture, contains unfilled `[TODO]` result placeholders. |
| `tiered_cache_paper.md` | Two-tier (VRAM + pinned-host) MoE expert caching with learned transition matrices — precursor to `markov_expert_prediction_eval.md`. |

## Provenance & retrieval research

| Doc | Covers |
|---|---|
| `bdp_calibration.md` | Per-layer depth-weight calibration for the MH_XOR_QQ_l0xl4 BDP retrieval signal across three attention-depth bands. |
| `coarse_to_fine_provenance_scan.md` | **Negative result.** Investigated a two-stage branch-and-bound sublinear scan; concluded the flat single-stage `BdpScanner` is already the right architecture — no lossy method beat it. |
| `provenance_strategy_results.md` | Comparison of attentional provenance strategies on Qwen3-30B-A3B across 48 tool-selection probes, with raw captured K/Q vector data. |
| `tool_selection_provenance_ideas.md` | Signal-extraction game plan for model-free tool selection from persisted provenance data; early-stage measurements (Top-1 ~3.9%) before the shipped design in `tool_selection_provenance_results.md`. |
| `tool_provenance_results.txt` | Raw harness output log backing `tool_selection_provenance_ideas.md`'s reference-scorer numbers. |

## KV cache & quantization kernel research

| Doc | Covers |
|---|---|
| `arena-compact-kernel-design.md` | GPU compaction pass to physically relocate live GID slots out of sparse arenas so they can be freed (draft). |
| `kernel_cache_design.md` | Mapping CUDA grid dimensions to L2 cache boundaries for quantized GEMM (Q4_K) on Ada Lovelace, to fix cache-thrashing at large batch sizes. |
| `q8_matmul_pipeline.md` | Tensor-core INT8 matmul path treating activations as a first-class 8-bit interchange format, with the FP residual stream as the error-compounding break. |
| `quantization_findings.md` | Extended research summary of KV-cache quantization error sources with numerical error-bound models. |
| `quantize-kernel-rewrite-design.md` | API draft for a unified Palette4-first quantize/transpose kernel replacing the earlier per-format kernel family. |
| `quantized_8bit_kv_cache.md` | F8_0/F8_1 FP8 (E4M3) KV quantization formats, extending the integer Q4/Q8 family for native FP8 tensor-core ops on sm_89. |
| `quantized_kv_cache.md` | v5.0 "Adaptive Block-Oriented Quantization" — the INT4/INT8 MMA pipeline and token-oriented KV cache design predating the current chunked-KV architecture. |
| `sealed_sequence_tiers.md` | Four-tier `SealedSequence` storage model driven by VRAM pressure and restoration demand — predates the current hot/warm/cold tier design in `kv_tier_migration.md`. |

## Kernel & pipeline designs superseded by later kernels

| Doc | Covers |
|---|---|
| `boundary_injection_design.md` | Static chunk cache for structural ChatML boundary tokens (system/user/assistant transitions), to skip repeated prefill of fixed inter-turn text. |
| `expert_pipeline_dataflow.md` | Async fork-join MoE expert pipeline design: CPU as a pure submission engine, GPU as a stream-dependency dataflow machine. |
| `paged_glue_kernel.md` | Earlier dedicated glue-attention kernel derived from paged-decode; superseded by the batched `GAP_FILL` approach in `docs/glue_prefill_kernel.md`. |
| `prefill_optimization.md` | **Built, 5.2–18.6× measured.** INT8 prefix-attention prefill kernel with GQA-packed tiles — the design record for what became `paged_prefill_int8_kernel.cuh`. |

## Conversation substrate & NPC design (early)

| Doc | Covers |
|---|---|
| `coding_assistant_conversation.md` | A flattened ChatML trunk simulating the Zen Code phased-ingestion pipeline as one linear conversation, for trunk-conversation experimentation. |
| `cognitive_architecture.md` | "The Three-Part Mind" — an NPC cognitive architecture split into Sleep, Daydreaming, and Reasoning processes over a static Beliefs substrate; precursor to `docs/npc_mind_design.md`. |
| `infinite_conversations.md` | v1 design for unbounded conversation history via a self-balancing AVL summary tree with a `dirty`-bit regeneration sweep. §7's tree structure is superseded by `docs/immutable_summary_forest.md`. |
| `summarization_design.md` | Early wiring plan for the `ConversationTree` summarization trigger, before the inference path for `run_summarize()` was implemented. |
| `time_division_memory_tree.md` | v2 fixed-budget summary tree for unbounded NPC life recall, with per-level compression ratios (structural routing → era → period → leaf detail). |
| `tree_gen_design.md` | Design for `tree_gen`, the guide-pipeline life-timeline generator that plans scaffolding before narrating to the character; see the shipped example at `candle-conversation/examples/tree_gen.rs`. |
