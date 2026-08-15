# candle-transformers

Model implementations and the fork's batched multi-session inference engine —
the layer that sits between raw tensor ops (`candle-core`, `candle-nn`) and the
`candle-conversation` substrate / `zend` daemon that consume it.

## What it does

`candle-transformers` has two halves. The first is a large, mostly-upstream
**model zoo**: `src/models/` holds well over a hundred architectures spanning
causal LLMs (Llama, Mistral, Mixtral, Gemma/Gemma2/Gemma3, Phi/Phi3, Qwen2,
Qwen3, Qwen3-MoE, DeepSeek2, GLM4, StarCoder2, Falcon, StableLM, RWKV, Mamba,
Persimmon, Yi, MPT, GraniteMoeHybrid, …), quantized (GGUF) variants of many of
them (`quantized_*.rs`), embedding/encoder models (BERT, DistilBERT, JinaBert,
ModernBERT, NvEmbed), vision models (ViT, DINOv2, ConvNeXt, EfficientNet,
SegmentAnything, DepthAnythingV2), vision-language models (CLIP, SigLIP,
LLaVA, Pixtral, PaliGemma, Moondream, BLIP), diffusion models (Stable
Diffusion, Flux, MMDiT), and audio models (Encodec, Mimi, DAC, Whisper-style
Metavoice/ParlerTTS, CSM). Most of these run through candle's standard
`Module`/`VarBuilder` forward path and are unmodified from upstream.

The second half is fork-specific and is the actual production path: **batched,
paged, continuous-batching inference** for the unbounded-context engine.
Instead of one sequence at a time, a `BatchedInferenceSession` drives many
sessions through shared per-layer weights with per-sequence paged KV caches,
adaptive KV quantization, and — for MoE models — a Markov-predicted expert
cache that keeps expert weight streaming ahead of the batch.

## Key modules / layout

| Path | Role |
|------|------|
| `src/models/*.rs` | The model zoo (see above); each file is one architecture. |
| `src/models/batched_inference.rs` | `BatchedInferenceSession`, `BatchedConfig`, `InferenceMode` (float/uniform-quant/adaptive C0–C10) — owns the chunked KV backing and per-sequence state (create/fork/free, glue-gap reservation, quantize-and-seal). |
| `src/models/batched_model.rs` | `BatchedModelCore` trait (model accessors) and `BatchedInference<M>` wrapper — owns RoPE caching and `forward_wave_contexts`, the single re-entrant forward that packs decode+prefill+glue rows into one activation buffer and layer range. |
| `src/models/batched_layer.rs` | Per-layer batched attention dispatch: `BatchedAttentionLayer`, mixed decode/prefill/glue group forwarding. |
| `src/models/expert_lre/` | The MoE expert pipeline: background-thread expert cache with mmap→VRAM DMA, score-based eviction, and Markov transition-based prefetch (see below). |
| `src/models/prefill_utils.rs` | Paged-prefill entry points (`paged_prefill_batched`, `paged_prefill_flat`) — the custom INT8/BF16 paged-attention prefill kernels, with a per-sequence CPU/flash-attn fallback for non-chunked caches. |
| `src/models/kv_cache_utils.rs`, `kv_collect_utils.rs`, `causal_mask_cache.rs`, `rope_tables.rs` | Shared batched-inference plumbing: `KvCaches`/`SequenceContext`, RoPE table precomputation, causal mask caching. |
| `src/models/batch_test/` | Fixtures and helpers for integration tests: `story.md`/`system.md`/`names.md` prompts, a captured Qwen3-30B-A3B routing trace, RULER-style long-context generators (`ruler_gen.rs`), HF-download test helpers. |
| `src/generation/` | `LogitsProcessor` and `Sampling` (argmax, temperature, top-k, top-p, top-k-then-top-p, Gumbel-softmax) — token sampling shared by every model driver. |
| `src/pipelines/` | Thin end-user pipeline scaffolding (text generation). |
| `src/quantized_nn.rs`, `src/quantized_var_builder.rs` | `Module` impls and `VarBuilder` backed by GGUF-quantized tensors, used by the `quantized_*` model variants and the batched engine's int8/qmatmul path. |
| `src/object_detection.rs` | Bounding-box / NMS helpers for detection models (YOLO-style outputs, SAM). |

## Key types & entry points

A caller building the batched engine starts from:

- [`BatchedInferenceSession`](src/models/batched_inference.rs) — create with
  `new`/`new_with_backings`, then `create_sequence`/`fork_sequence`/`free_sequence`
  per conversation, `reserve_glue_gap`/`inject_sealed_at_tail` for
  provenance-selected reprojection, `quantize_and_seal_sequences` to run the
  adaptive KV compression kernel over a sequence's live chunks.
- [`BatchedConfig`](src/models/batched_inference.rs) / `InferenceMode` — choose
  KV storage: a fixed float/uniform-quant format, or an adaptive `C0`..`C10`
  compression level that engages `CompressionPolicy`'s per-block selection
  kernel (re-exported from `candle-nn::kv_cache`).
- [`BatchedModelCore`](src/models/batched_model.rs) — the trait a model
  implements (layer access, embeddings, head, RoPE convention) to plug into
  `BatchedInference<M>::forward_wave_contexts`, the single re-entrant forward
  used by every wave (decode, prefill, and glue rows, any contiguous layer
  range, resumable via its `WavePhase::Residual`/`Logits` return).
- [`LogitsProcessor`](src/generation/mod.rs) — samples a token from a forward's
  logits (`Sampling::{ArgMax, TopK, TopP, TopKThenTopP, GumbelSoftmax}`).

## The MoE expert pipeline (`expert_lre/`)

For Mixture-of-Experts models whose expert weights don't fit resident in VRAM
(e.g. Qwen3-30B-A3B on a 16 GB card), `expert_lre` streams experts from a
memory-mapped checkpoint into a fixed-size VRAM pool on demand. Its two
central mechanisms:

- **Markov expert prediction** (`transition.rs`, `TransitionMatrix`): an
  online-learned `[E × E]` co-occurrence matrix per adjacent-layer pair, scored
  by pointwise mutual information. Given the experts active at layer `L`, it
  predicts which experts layer `L+1` will need and starts their DMA before `L`
  finishes computing — converting cold misses into overlapped loads. Measured
  ~69% hit rate on Qwen3-30B-A3B; see `docs/markov_expert_prediction_eval.md`
  for the full offline evaluation (LOOCV promotion/eviction study) and
  `eval.rs` for the harness.
- **Wave-batched grouped GEMM** (`pipeline.rs`, `compute.rs`,
  `gpu_dispatch.rs`): many concurrent sessions are stepped through each layer
  together so one expert weight load is amortised across the whole batch
  rather than paid per session. `handle.rs` exposes `ExpertCache` with two
  modes — **threaded** (background pipeline thread owns cache state, used
  when experts stream from mmap) and **inline** (all experts pre-loaded,
  Mutex-protected, no DMA).

Eviction is score-based (frequency-decayed, layer-aware, with early-layer
pinning); see the module header in `expert_lre/mod.rs` for the full four-part
policy. `docs/gpu_native_moe_dispatch.md` covers the GPU-native routing
dispatch that removes the per-layer expert-routing GPU→CPU readback, and
`docs/unified_wave_inference_engine.md` / `docs/continuous_fair_waves.md`
cover how decode, prefill, and glue forwards are interleaved so decode's hot
expert working set survives large prefills.

## Primary models

| Model | Role |
|-------|------|
| Qwen3-30B-A3B | Current development/benchmarking target (MoE, 3B active of 30B total). |
| Qwen3-235B-A22B | Production Zen Code target (requires the 2× RTX 5090 workstation). |
| Llama-3.2-3B | `batch_test` integration testing (dense, cheap to iterate on). |
| Qwen3-8B/14B | Ablation baselines. |

KV quantization thresholds (`PRODUCTION_K_QREL_*` / `PRODUCTION_V_QREL_*` in
`candle-nn`'s `CompressionPolicy`) are **model-specific** and must be
re-derived by measurement whenever a new model is added — they are not
portable across architectures or parameter counts.

## How it is used

`candle-conversation`'s scheduler and `zend`'s daemon are the callers: they
build a `BatchedInferenceSession`, wrap a model in `BatchedInference<M>` by
implementing `BatchedModelCore`, and drive `forward_wave_contexts` once per
wave with a layer range and packed decode/prefill/glue contexts. `candle-nn`
supplies the `ChunkedKvBacking`/`KvCache`/`CompressionPolicy` types this crate
builds sessions on top of.

Feature flags (mirrors `candle-core`/`candle-nn`): `cuda` (required for the
batched/paged production path), `cudnn`, `flash-attn` (pulls in
`candle-flash-attn` as an optional fast path for a handful of standalone,
non-batched model forwards — not the paged production path), `mkl`,
`accelerate`, `metal` (CPU-parity backend for the upstream model zoo), plus
crate-local `profile`, `huge-context`, and `ruler-bench` (enables the RULER
long-context generator's `tokenizers` dependency).

## Related docs

- `docs/markov_expert_prediction_eval.md` — the Markov Wave paper: promotion/eviction study and final design.
- `docs/unified_wave_inference_engine.md` — the original decode/prefill/glue wave design.
- `docs/continuous_fair_waves.md` — supersedes the above: decode and prefill share the layer traversal instead of time-slicing it.
- `docs/gpu_native_moe_dispatch.md` — GPU-native MoE routing dispatch (removes the per-layer GPU→CPU readback).
