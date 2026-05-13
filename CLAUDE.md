# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What We're Building

This is a **custom Candle fork** implementing an unbounded-context LLM inference engine. The core innovation is achieving **O(1) numerical error at any context depth** by combining:

1. **Provenance-selected attention** — attention over a retrieved relevant subset, not full sequential history
2. **Three-tier paged KV cache** — GPU (hot) → RAM (warm) → NVMe (cold), with async prefetch
3. **Adaptive per-block KV quantization** — 10 compression levels (C0–C9), format selected per 32-token block

This powers two products:
- **Zen Code** — a persistent AI coding assistant (`zend` daemon + `zen-vscode` Continue fork) with institutional memory across sessions and shared KV prefix across developer forks
- **Battle Cities** — an NPC narrative game with unbounded agent memory

**Target paper submission**: arXiv May 13, 2026 (canonical), then MLSys/ACL/EMNLP/NeurIPS 2027.

---

## Core Technical Concepts

### O(1) Error Theorem
Under provenance-selected attention (not full-sequence attention), total numerical error per step is **O(1)** independent of context depth. All design choices (two-phase quantization, attention sink protection, Q4_KS/Q8_KS formats) serve this property.

### Adaptive Quantization (C0–C9)
Every 32-token block is independently evaluated. The `CompressionPolicy` selects K/V formats per block based on cosine distance thresholds:
- **C0** — near-lossless (K: R16/F16 fallback, V: Q8_0). Reference quality.
- **C4–C5** — moderate compression (K: Q8_0/Q8_KS, V: Q4_1/Q4_KS)
- **C9** — maximum compression (K/V predominantly Q2_0)

**Asymmetric K/V**: Keys are sensitive by channel, Values by token. Separate threshold tables exist for each (`PRODUCTION_K_QREL_*`, `PRODUCTION_V_QREL_*`).

**Attention sink protection**: The first 4 tokens (positions 0–3) use a dedicated fine scale (Q4_KS/Q8_KS) to prevent global scale inflation from attention sink magnitudes.

### Attentional Provenance Retrieval
Q vectors are captured live during decode (not pre-computed from embeddings). A flat CPU scan over all KV chunks uses Binary Directional Provenance (sign(Q_PCA^T @ K), XOR+popcount with VNNI) completing in 3–10 ms across the full unbounded corpus. Results prefetch from warm/cold tiers while GPU decodes the next token.

### Markov Expert Prediction (MoE models)
Routing patterns from the prior layer predict the current layer's expert needs. Measured 69% hit rate on Qwen3-30B-A3B. Wave-batched grouped GEMM steps 64 concurrent sessions coherently through layers, amortising expert weight loads across sessions.

---

## Primary Models

| Model | Use | Notes |
|-------|-----|-------|
| **Qwen3-30B-A3B** | Current development/benchmarking | 30B total, 3B active, MoE |
| **Qwen3-235B-A22B** | Production Zen Code target | Requires RTX 5090 workstation |
| **Llama-3.2-3B** | batch_test integration testing | VibeStudio/Nidum uncensored fine-tune |
| Qwen3-8B/14B | Ablation baselines | — |

Qwen3 thresholds are model-specific and must be re-derived for each variant. When a new model is added, re-derive the `PRODUCTION_*` constants via measurement.

---

## Hardware

**Current dev machine** — RTX 4090 Mobile 16 GB GDDR6.
- Benchmarks: 509 t/s single-session, 2,446 t/s aggregate (64 sessions), Qwen3-30B-A3B.

**Ordered production workstation** (~mid-2026):
- 2× RTX 5090 32 GB GDDR7 (Blackwell, water-cooled)
- AMD Threadripper 7970X 32C/64T (AVX-512 for provenance scan)
- 512 GB DDR5-5200 ECC (warm tier KV cache)
- 16 TB PCIe 5.0 NVMe RAID 0 @ 45 GB/s (cold tier)

---

## Build Commands

```bash
# Basic build/check
cargo check --workspace
cargo build --workspace --release

# Feature-gated (most production code requires cuda)
cargo build --features cuda
cargo build --features cuda,cudnn

# Testing
cargo test                          # CPU only
cargo test -p candle-core           # single crate
cargo test -p candle-nn kv_cache    # single module
cargo test --features cuda          # GPU tests
cargo test --release

# Linting (enforced in CI)
cargo fmt --all -- --check
cargo clippy --workspace --tests --examples -- -D warnings

# Force full CUDA kernel recompilation
make clean-ptx
```

---

## Crate Architecture

```
candle-examples / candle-conversation
       ↓
candle-transformers   (model impls, batched inference, Zen Code/Battle Cities models)
       ↓
candle-nn             (layers, VarBuilder, KV cache system ← most active work)
       ↓
candle-core           (Tensor, Device, DType, ops)
       ↓
candle-kernels        (AOT CUDA kernels: paged-decode/, paged-prefill/, quantized/)
```

**candle-core** — `Tensor`, `Device` (Cpu/Cuda/Metal), `DType`, op dispatch. Feature-gated backends in `cpu_backend/`, `cuda_backend/`, `metal_backend/`.

**candle-kernels** — NVCC-compiled with SHA256-based caching in `build.rs`. Key subdirs: `paged-decode/`, `paged-prefill/`, `quantized/`, `simple/`. `CHUNK_SIZE = 32` is the shared Rust/CUDA constant. PTX embedded at compile time.

**candle-nn** — The most actively developed crate. `kv_cache/` is where all the compression/paging/arena work lives.

**candle-transformers** — `batched_inference.rs` + `batched_model.rs` are the high-level batched inference API. `batch_test/` has story/system prompt fixtures for integration tests.

---

## KV Cache Subsystem (`candle-nn/src/kv_cache/`)

The most complex part of the codebase. Key files:

| File | Role |
|------|------|
| `mod.rs` | Public API, `KvFormat`, `QuantFormat`, `PagedKvArenas` trait |
| `cache.rs` | `Cache`, `KvCache` (contiguous, simple baseline) |
| `rotating.rs` | `RotatingKvCache`, `ScatteredKvCache` |
| `arena_table.rs` | `ArenaTable`, `PerHeadTable`, palette indexing |
| `chunked/mod.rs` | `ChunkedKvBacking`, exports, constants |
| `chunked/arena.rs` | `Arena` enum (Float/Quantized), `StoragePolicy` |
| `chunked/compress.rs` | Per-block format selection, quantization kernel calls |
| `chunked/compression_policy.rs` | `CompressionPolicy`, level→candidate tables, production thresholds |
| `chunked/eviction.rs` | `EvictionManager`, warm/cold tier decisions |
| `chunked/alloc.rs` | `GidPool`, free-list chunk allocator |
| `chunked/warm_pool.rs` | RAM-tier dense packing |
| `chunked/gpu_chunks.rs` | GPU-side arena storage |

**`KvFormat`**: `Float(DType)` or `Quantized(QuantFormat)`. All quant blocks = 32 elements.

**`CompressionPolicy`**: Given a level (C0–C9), returns ordered candidate lists for K and V. Evaluation in `compress.rs` picks smallest format passing cosine distance threshold.

---

## `VarBuilder` Pattern

All weight loading uses `VarBuilder`:
- `from_mmaped_safetensors` — memory-mapped SafeTensors (preferred for large models)
- `from_gguf` / `quantized_var_builder` — GGML quantized checkpoints
- `zeros` / `from_varmap` — for training/init

---

## Adding a CUDA Kernel

1. Add `.cu` to `candle-kernels/src/<subdir>/`
2. Register in the archive group in `candle-kernels/build.rs` (SHA256 triggers recompile)
3. Add FFI binding in `candle-kernels/src/lib.rs`
4. Call from `candle-core/src/cuda_backend/` via `unsafe`

---

## Platform Notes

- **Windows + CUDA**: `cuda.dll`, `cublas.dll`, `curand.dll` must be on PATH.
- **WSL**: Never load models from `/mnt/c` — I/O is extremely slow; copy to native Linux FS.
- **Flash attention**: requires `git submodule update --init` and `--features flash-attn`.
