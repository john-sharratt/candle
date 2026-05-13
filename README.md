# One Card, One Stack

**Constraint-Driven Architecture for Asymptotically Stable Inference over Unbounded Agent Memory**

> Research fork of [Hugging Face Candle](https://github.com/huggingface/candle).
> Full technical report: [docs/unbounded_agents.md](docs/unbounded_agents.md)
> Target submission: arXiv May 2026 → MLSys / ACL / EMNLP / NeurIPS 2027

---

## The Problem

Persistent agentic systems require context that grows without bound. Under standard full attention, numerical error per generation step grows monotonically with context depth — for any finite-precision arithmetic, any compression scheme, on any hardware — because every token participates in every subsequent computation with equal structural weight. This is not a compression problem; it is an architectural one. More VRAM defers the threshold; it does not eliminate the structural problem.

## The Fix

**Theorem (§11.2 — Asymptotic Numerical Stability):** Under provenance-selected attention over a tiered context, total numerical error per generation step is bounded by a constant **O(1) independent of context depth N**, in contrast with the O(N) scaling of standard full-attention systems.

The fix is architectural: decouple the set of tokens participating in any generation step from the total number of tokens in context. Something functionally equivalent to provenance-selected sparse attention is *necessary* — not sufficient, but necessary — for a persistent-session LLM to maintain bounded quality as context grows. No full-attention system, on any hardware, can provide the same guarantee.

## Benchmarks

Tested on a single **RTX 4090 Mobile (16 GB GDDR6)** running **Qwen3-30B-A3B**:

| Workload | Throughput |
|---|---|
| Single persistent session | **509 t/s** |
| 64 concurrent persistent sessions | **2,446 t/s aggregate** |

For comparison, community benchmarks report 150–195 t/s for this model on RTX 4090 24 GB with Ollama/llama.cpp (single session). No standard framework runs this model on 16 GB at 64-session concurrency.

---

## System Contributions

### 1. Online Markov Expert Prediction with Wave-Batched MoE

A self-learning transition matrix predicts expert routing from actual inference observations, with no offline calibration. Combined with a wave-batched grouped GEMM kernel that coalesces expert work across concurrent requests, it achieves stall-free MoE inference under partial VRAM residency.

- **69% hit rate** on Qwen3-30B-A3B routing prediction
- DMA offload for expert weight streaming during decode
- 64 concurrent sessions advanced coherently through layers in a single wave

### 2. Three-Tier Paged Context

A paged context architecture spanning three storage tiers:

```
GPU VRAM (hot)  →  CPU RAM (warm)  →  NVMe (cold)
```

Blocks are independently managed at 32-token granularity. The adaptive quantization system selects per-block K/V formats based on cosine-distance thresholds computed at seal time. Two-phase prefill refresh at turn boundaries eliminates the autoregressive numerical drift that degrades generation quality beyond ~500 decode steps.

The Asymptotic Numerical Stability theorem is proven over this architecture: under provenance-selected attention with hot-tier blocks originating from prefill-refreshed activations, the system error floor approaches the hot-tier rounding constant, independent of how many blocks reside in warm or cold tiers.

### 3. Attentional Provenance Indexing with Speculative Context Decode

**Provenance indexing:** Q vectors are captured live during decode as persistent cognitive-state fingerprints. A flat CPU scan over all KV chunks uses Binary Directional Provenance (sign(Q_PCA^T @ K), XOR + popcount with VNNI), completing in **3–10 ms** across the full unbounded corpus regardless of context depth.

**Speculative Context Decode:** A pipelined two-session generation loop hides the CPU provenance scoring behind a parallel variable-window probe session (up to 64 tokens, terminated at newline boundaries). The probe session's Q/K fingerprints drive CPU provenance scoring at each reasoning step boundary, assembling the next context window while the current one decodes. Probe tokens are discarded and never enter the KV cache.

**Evaluation:** The system ingests its own 2.2M-line Rust/CUDA Candle fork via a ~20M-token learning-phase conversation, then retrieves via iterative multi-hop dependency analysis during decode. The one-shot ablation (same index, single pre-generation retrieval) isolates the contribution of continuous decode-time retrieval — iterative retrieval discovers transitive dependencies that pre-generation retrieval misses, with accuracy independent of dependency chain depth.

### 4. Native Quantized Inference Stack

Standard GEMM libraries dequantise weight matrices to full precision before computation, which OOMs during prefill on 16 GB. This stack writes native quantized matmul kernels that never materialise a full-precision weight copy.

- 720+ CUDA kernels compiled AOT with SHA-256 change detection
- Paged decode and prefill kernels operating at 32-token chunk granularity
- Fused sampling kernel covering all common sampling modifiers in one CUDA launch
- Greedy decomposition for smooth 1–500 token throughput without remainder handling
- Vectorised scalar CPU fallbacks for all quantised formats

---

## Applications

### Zen Code

A persistent AI coding assistant with genuine long-term memory across sessions. Architecture:

- `zend` — background daemon hosting the conversation engine
- `zen-vscode` — VS Code extension (Continue fork) consuming the gRPC API
- Shared KV prefix across developer workspace forks
- Institutional memory: facts, decisions, and code relationships accumulate across all sessions

### Battle Cities

An NPC narrative game where each agent maintains unbounded memory across story branches. The conversation tree architecture supports branching, summarisation, and task nodes with continuous KV projection.

---

## Crate Architecture

```
candle-conversation        (conversation engine, ~50 KLOC)
       ↓
candle-transformers        (batched inference, MoE models)
       ↓
candle-nn                  (layers, VarBuilder, KV cache system)
       ↓
candle-core                (Tensor, Device, DType, ops)
       ↓
candle-kernels             (AOT CUDA kernels)
```

### candle-kernels

NVCC-compiled with SHA-256-based caching. Key subdirs: `paged-decode/`, `paged-prefill/`, `quantized/`, `simple/`. `CHUNK_SIZE = 32` is the shared Rust/CUDA constant.

### candle-nn — KV cache subsystem

The most actively developed crate. `kv_cache/` contains the compression, paging, and arena subsystems.

### candle-transformers

`batched_inference.rs` + `batched_model.rs` are the high-level batched inference API. `batch_test/` has story/system prompt fixtures for integration tests.

### candle-conversation

Complete multi-session inference server. Key subsystems: session scheduler, conversation tree, substrate projection, provenance store, narrator/streaming bridge.

---

## Build

```bash
# Basic check
cargo check --workspace

# Release build with CUDA
cargo build --workspace --release --features cuda

# Run tests
cargo test --workspace
cargo test --features cuda          # GPU tests

# Linting
cargo fmt --all -- --check
cargo clippy --workspace --tests --examples -- -D warnings
```

**Windows:** `cuda.dll`, `cublas.dll`, `curand.dll` must be on PATH.

---

## Hardware

**Current dev machine:** RTX 4090 Mobile 16 GB GDDR6

**Target production workstation (~mid-2026):**
- 2× RTX 5090 32 GB GDDR7 (Blackwell)
- AMD Threadripper 7970X 32C/64T (AVX-512 for provenance scan)
- 512 GB DDR5-5200 ECC (warm-tier KV cache)
- 16 TB PCIe 5.0 NVMe RAID 0 @ 45 GB/s (cold tier)

---

## Primary Models

| Model | Role |
|---|---|
| Qwen3-30B-A3B | Current development target |
| Qwen3-235B-A22B | Production Zen Code target (RTX 5090 workstation) |
| Llama-3.2-3B | Integration test baseline |

---

## License

This repository is a fork of [Hugging Face Candle](https://github.com/huggingface/candle), which is dual-licensed under [MIT](LICENSE-MIT) and [Apache 2.0](LICENSE-APACHE). All modifications in this fork are released under the same dual license.
