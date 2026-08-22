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
Q vectors are captured live during decode (not pre-computed from embeddings). A paged GPU BDP scan over the resident gallery arena (Binary Directional Provenance: sign(Q_PCA^T @ K) agreement via IMMA/BMMA kernels; `candle-conversation/src/provenance/gallery_arena/scan.rs`) completes in milliseconds across the full unbounded corpus. Results prefetch from warm/cold tiers while GPU decodes the next token.

### Markov Expert Prediction (MoE models)
Routing patterns from the prior layer predict the current layer's expert needs. Measured 69% hit rate on Qwen3-30B-A3B. Wave-batched grouped GEMM steps 64 concurrent sessions coherently through layers, amortising expert weight loads across sessions.

---

## Primary Models

| Model | Use | Notes |
|-------|-----|-------|
| **Qwen3-30B-A3B** | Current development/benchmarking | 30B total, 3B active, MoE |
| **DeepSeek-V4-Flash-0731** | Native-sparse 1M-context port (in progress) | 284B total, 13B active, MXFP4 experts, K≡V latent attention; see `docs/deepseek_batched_paged_attention_plan.md` |
| **Qwen3-235B-A22B** | Production Zen Code target | Requires RTX 5090 workstation |
| **Llama-3.2-3B** | batch_test integration testing | VibeStudio/Nidum uncensored fine-tune |
| Qwen3-8B/14B | Ablation baselines | — |

Qwen3 thresholds are model-specific and must be re-derived for each variant. When a new model is added, re-derive the `PRODUCTION_*` constants via measurement.

---

## Hardware

**Current dev machine** — RTX PRO 5000 Blackwell 72 GB (sm_120).
- Reference benchmarks (measured on the previous RTX 4090 Mobile 16 GB): 509 t/s single-session, 2,446 t/s aggregate (64 sessions), Qwen3-30B-A3B.

**Ordered production workstation** (~mid-2026):
- 2× RTX 5090 32 GB GDDR7 (Blackwell, water-cooled)
- AMD Threadripper 7970X 32C/64T (AVX-512 for provenance scan)
- 512 GB DDR5-5200 ECC (warm tier KV cache)
- 16 TB PCIe 5.0 NVMe RAID 0 @ 45 GB/s (cold tier)

---

## Code Conventions & Engineering Principles

These apply repo-wide. They are deliberate standing decisions, not suggestions.

- **No backward compatibility.** This is a pre-publication research codebase. It is fine — expected — to break everything before you. Do not write compatibility shims, dual code paths, or `Option`-typed feature flags that exist only to keep an old path alive. Replace the real thing. (Genuine `Option`s — a real hit/miss, a value that is legitimately absent — are fine; optionality-as-a-feature-flag is not.)
- **No environment-variable feature flags. Ever.** Never gate a code path, optimization, or behavior on `std::env::var(...)` / an env toggle. When a new path is correct, make it *the* path — replace the old one in the code. When it is not correct yet, do not land it. An env flag to "keep the old path alive / opt in to the new one" is a dual code path by another name and is prohibited exactly like the backward-compat shims above. (Reading genuine deployment config — a model path, a device id — is not a feature flag and is fine.)
- **No `TODO`s, no stubs.** Never commit `TODO` / `FIXME` / `unimplemented!()` / `todo!()` / placeholder stubs. Implement the task fully or do not commit it. "Fail forward" means solve the hard thing and move on — it never means leaving a placeholder.
- **NEVER defer work — always finish the full task.** "Deferred to later", "punted for a future pass", "left for next session", "good enough for now" are not acceptable framings. If the task includes cleanup, finish the cleanup. If it includes a TODO comment, replace it with the implementation. If it includes a known regression to fix, fix it before declaring done. A task is done when every part of it is done, not when the interesting part is done. Pushing the boring tail of a task into the future means it never gets finished — past me always thinks future me will care, future me never does.
- **Imports, not fully-qualified paths.** Never write a fully-qualified type path inline (`crate::foo::Bar`). `use`-import every type at the top of the file and refer to it by its short name.
- **One concern per file.** Split modules into a subfolder with a file per concern rather than growing one large file. Prefer small, independently-testable units.
- **TDD, extensive unit tests.** Build tests alongside the code, as the code is written — not after. Every building block must be testable in isolation. For serialization / quantization / codec code, assert against **raw expected bytes**, never error-tolerance thresholds.
- **Design docs are authoritative.** When a design document exists for the work (e.g. `docs/*.md`), it takes precedence over discrepancies with the code. If the document is itself wrong, fix the document in the same change.
- **Code comments describe the implementation, not the design process.** Write every comment as if the full final design is already in place. No "Phase 2 of …", no "reserved for Phase 3", no "until Phase 4 lands", no "pre-Phase-N path", no "later phases will pivot on this". The design doc lives in `docs/`; code comments explain *what this code does and why* in the present tense, against the codebase as it is. A reader who has never seen the rollout plan should be able to understand the comment.
- **Persistence is mandatory.** The conversation substrate is always backed by its on-disk persistence layer (`candle-conversation/src/persistence/`, redo log at `.substrate/substrate.log`). There is no in-memory-only substrate mode. See `docs/kv_tier_migration.md`.
- **Never `git commit` without explicit permission.** Show the diff (or summarize what will be staged + propose the message) and wait for the user to say go. Authorization for one commit does not carry forward to subsequent commits — every commit requires its own approval. This is non-negotiable.
- **Only the `Edit` and `Write` tools may modify files.** Never rewrite a file from an external process — not PowerShell, not a Python script, not `sed -i`, not a shell redirect. There is no size or repetition threshold that justifies it; for a repeated pattern use `Edit` with `replace_all`, and for genuinely distinct sites issue several `Edit` calls in one message. Three separate reasons, any one of which is sufficient:
  - **A script fails silently.** `s.replace(a, b)` that matches nothing is a no-op, so a stale or mistyped pattern applies part of a multi-site change and reports success. `Edit` verifies against the file it read and errors when the match is missing or ambiguous, which is the difference between finding out now and finding out from a compile error three steps later.
  - **A script destroys on interruption.** `open(path, 'w')` truncates before it writes. If the machine dies in that window the file is gone rather than merely unchanged — and on NTFS the size metadata can land while the data does not, leaving a file of exactly the right length filled with `NUL`. That happened: a hard reset during this work left `candle-transformers/src/models/batched_model.rs` as 32,689 zero bytes, and ~6.7 KB of uncommitted changes had to be reconstructed from call sites. Editor writes leave the file untouched on failure.
  - **A script is opaque.** Tool edits are tracked, so "what touched this file" is a lookup. Script writes are not, so the same question becomes forensics.
  - PowerShell has an extra failure of its own on top: `Set-Content`, `Out-File`, and `[System.IO.File]::WriteAllText` default to encodings that mangle UTF-8 multi-byte content via a CP1252 round-trip — em-dashes (`—`), box-drawing characters (`┌─└`), arrows (`→`, `≤`, `≥`), curly quotes, and anything else outside ASCII silently become mojibake like `â€"` or `â”Œâ”€`, hard to spot at edit time and easy to ship to commit. Even `-Encoding utf8` doesn't fix it, because the read side may already have decoded the file as CP1252.
  - External processes are fine for **reading** state (`grep`, `sed -n`, `git show`) and for running builds and tests. The prohibition is on writes.
- **Never mask a command's exit status.** `cmd | grep -E "^error"`, `| head`, `| tail`, and `cmd; echo DONE` all report success for a command that failed — `head` can `SIGPIPE` the writer mid-output, and a `;` chain runs the echo unconditionally. A build or test result read through a filter is not evidence. Let the command's own exit code decide, and when filtering output for brevity, check the status separately (`cmd > log 2>&1; echo "EXIT=$?"`, then read the log).

---

## Hot-Path Invariants (Prefill & Decode)

The batched inference loop (`forward_wave` and everything it calls per layer, per wave)
must satisfy these six invariants. They are the standing target that turns the current
single-session rate into the compute-bound ceiling (**prefill ≥ 1000 t/s, decode ≥ 50 t/s**).
Every violation is a place the GPU sits idle, round-trips through the host, or does work the
architecture says is unnecessary. Full study + per-invariant violation catalogue with
`file:line` references lives in `docs/deepseek_hot_path_invariants.md` (authoritative).

1. **No `to_dtype` in the loop — kernels emit the final type.** Every dtype conversion on
   the hot path is a full-tensor memory pass a kernel could have avoided by writing its
   output in the type the next consumer wants. Norms emit the kernel's input type; attention
   kernels emit the out-proj's input type.
2. **No allocate-plus-copy to materialise a layout, by any spelling.** `contiguous()`,
   `force_contiguous()`, `Tensor::cat`, and `slice_set` are the SAME operation as far as this
   invariant is concerned — each allocates and copies so a consumer can be handed the layout it
   prefers, and `cat`/`slice_set` cost **one launch per argument**. If a consumer needs a layout,
   teach it to read the layout that exists (offset + stride, or a descriptor table — see 2b), or
   produce that layout directly from the kernel that made the data.
   > This invariant was originally worded as "no `contiguous` / `force_contiguous`", naming two
   > functions rather than the operation. `cat` and `slice_set` matched neither name and were
   > never audited: a measured 892,104 of 1,079,568 copy launches per sweep — 2.5% of GPU — sat
   > entirely outside the rule. Police the operation, not the spelling.
2b. **A kernel consuming per-session or per-row data takes a DESCRIPTOR TABLE, not a packed block.**
   Requiring one dense base pointer is what forces the caller to `cat`/`slice_set` rows together,
   so the copy is the kernel's API bug, not the caller's. Pass a device table of
   `{ptr, offset, stride, len}` per row/session and read in place. The pattern already exists here:
   `candle-kernels/src/arena_table.cuh` (`ArenaTableEntry`/`PerHeadTableEntry`) for the paged
   attention kernels, the gallery's `region_ptr_cache`, and `bdp_recall_batched`'s per-gallery
   sign-pointer table (which replaced an O(Σ len × words) concatenation).
3. **No unnecessary GPU→CPU transfers.** Exactly two sanctioned readbacks: (a) MoE expert
   routing (`indices` → host), because the streaming `ExpertCache` schedules pinned→VRAM
   uploads by expert id; and (b) the embedding lookup (token ids → host, CPU `index_select`,
   transfer in), a pure index + transfer that keeps the embed table off VRAM. Everything else —
   comp-idx assembly, select remaps, gather indices — stays on the GPU.
4. **Run as much as possible on the GPU.** No host-side compute a kernel can do: no host
   counting-sort, no host set-union/dedup/remap. Host code issues launches; it does not
   compute over per-token data. (The two transfers in #3 are the only exceptions.)
5. **Everything in prefill and decode runs fully batched.** One launch over all slots / all
   sessions, not a per-seq or per-token loop. Decode batches select, gather, attention, and
   out-proj across sessions; prefill must reach the same shape — a multi-slot attention kernel
   over all prompt slots, not one launch per sequence.
6. **Never zero memory in the inference loop — it will be written anyway.** A buffer a kernel
   fully overwrites must be allocated uninitialised (`alloc_uninit`), never `Tensor::zeros`
   (a second full-width `memset` on the exact bytes the kernel is about to stamp). The only
   buffers that may be zeroed are ones whose zero value is read before being written: atomic
   accumulators (`atomicAdd`/`atomicMax` targets), scatter bases, and ragged padding.

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
| `arena_table.rs` | `ArenaTable`, `PerHeadTable`, `ArenaLocation`, palette indexing |
| `chunked/mod.rs` | Exports, constants |
| `chunked/backing.rs` | `ChunkedKvBacking` — core cache state |
| `chunked/arena.rs` | `Arena` enum (Float/Quantized), `ArenaKey`, `StoragePolicy` |
| `chunked/gid_pool.rs` | `GidPool`, `ChunkGid` — free-list chunk allocator |
| `chunked/head_gids.rs` | `HeadGids` — per-head/palette chunk GID collection |
| `chunked/compress.rs` | Per-block format selection, quantization kernel calls |
| `chunked/compression_policy.rs` | `CompressionPolicy`, level→candidate tables, production thresholds |
| `chunked/sequence_ops.rs` | Sequence injection / sealing operations |
| `chunked/io.rs` | Contiguous read/write, raw sealed-chunk extraction |
| `chunked/gpu_chunks.rs` | GPU-side arena storage, kernel table builders |
| `chunked/types.rs` | `SealedSequence`, `SealedChunk`, `ChunkMeta` |

> **Tiering status:** the three-tier KV cache is **built and wired** — GPU (hot)
> → RAM (warm, CPU arenas) → NVMe (cold, append-only redo log). hot→warm runs on
> the persistence thread (`migrate_group_hot_to_warm` → `migrate_sealed_to_cpu_batch_async`),
> warm→hot on demand (`elevate_to_hot`), cold is the redo log at `.substrate/substrate.log`.
> Two divergences from the `docs/kv_tier_migration.md` target remain: warm residency
> is *pageable* CPU arenas (not the doc's pinned `warm_pool.rs`, which was never built —
> so warm↔hot runs at ~½ PCIe bandwidth), and the hot→warm copy runs on the primary
> stream (no dedicated overlap stream). See `docs/kv_tier_migration.md` for the design.

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
