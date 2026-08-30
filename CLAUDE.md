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

**Three dev machines, worked on in parallel.** The framework must work on all
three, and none is "the" target — check what you are actually on (`nvidia-smi`)
rather than assuming, because sizing decisions differ between them:

| Machine | VRAM | SM | Host | Link |
|---|---|---|---|---|
| **RTX 4090 Mobile** | 16 GB | sm_89 (Ada) | 32 GB RAM | PCIe 4.0 ×16, ~25 GB/s |
| **RTX 3090** | 24 GB | sm_86 (GA102) | i7-10700K 8C/16T, 64 GB RAM | **PCIe 3.0 ×16, ~12 GB/s** |
| **RTX PRO 5000 Blackwell** | 72 GB | sm_120 | — | PCIe 5.0 ×16 |

Two traps specific to the **3090 box**, both easy to miss because the card
looks like the biggest consumer part in the fleet:

- **Its host caps the link at PCIe 3.0.** The i7-10700K is Comet Lake, so
  host↔GPU is ~12 GB/s — roughly *half* the 4090 Mobile's, despite 50% more
  VRAM. Any sizing that reuses the "~25 GB/s" figure from
  `docs/expert_cache_design.md` is wrong there by 2×. That table now names its
  machine per row; keep it that way.
- **sm_86 has no native FP8** (`device_caps.cuh` gates `has_native_fp8` at
  `sm >= 890`). The sm_89 b1 BMMA and Blackwell INT8 IMMA provenance backends
  also do not apply; both degrade to the next rung, so it runs — just not on
  the fast path. Do not benchmark provenance scan here and compare to the
  4090 Mobile without accounting for it.

At 24 GB the 3090 crosses `Q6_MIN_TOTAL_VRAM_BYTES`, so it runs
**Qwen3-30B-A3B Q6_K** (~25 GB, expert-LRU paged) where the 16 GB box runs
Q4_K_M — see `zend/src/model_choice.rs`.

Reference benchmarks (measured on the 4090 Mobile 16 GB): 509 t/s
single-session, 2,446 t/s aggregate (64 sessions), Qwen3-30B-A3B.

**Model size is not bounded by VRAM.** The three-tier expert cache streams
VRAM → pinned RAM → mmap, so a MoE model's resident footprint is its dense
weights plus whatever expert working set fits — which is why a 30B-A3B runs
on the 16 GB card. A bigger card buys speed, not feasibility. Never conclude
a model "cannot run here" from parameter count alone.

**Ordered production workstation** (~mid-2026) — a third target, not a
prerequisite for any phase:
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
1b. **VALIDATE the type, do not CONVERT it — `expect_dtype`, never a defensive `to_dtype`.**
   Where two types are *supposed* to agree, assert it with
   `models::operand_guard::{expect_dtype, expect_dense, expect_dense_dtype}` (layout metadata
   only — no allocation, no launch, free on the hot path). A `to_dtype` there is wrong whichever
   way it lands: when the types already match it is dead code that protects nothing while
   silently absorbing a producer that later starts handing over the wrong type, and when they
   do not it is a full-tensor pass per call, per layer, per step — invisible, because the line
   reads as a cast rather than as a copy. The same applies to a defensive `contiguous()`
   (invariant 2). If a genuinely different type or layout must be supported, teach the consumer
   to read it — a template parameter, a stride argument, a producer that emits the right type —
   rather than rewriting the tensor at the call site. Reserve `to_dtype` for conversions the
   design actually calls for (an F32 accumulator deliberately narrowed for storage, a table
   built once at load), and say in a comment which it is.
   > This is not hypothetical in either direction. The MTP capture buffers were allocated F32
   > against a BF16 wave, so a "harmless" cast at each end was a real launch per sequence per
   > wave; and the casts that *were* no-ops sat over exactly the mismatches that would otherwise
   > have been caught at the boundary that introduced them.
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
7. **The span's partition boundaries hold in BOTH directions, and every tenant re-checks the
   one it is about to cross.** The device reservation is one contiguous span shared by four
   tenants — `| persist | KV regions | wave transient tier | expert weights |` — with the
   *elastic* boundary `weight_floor` between the middle two. Nothing in CUDA enforces it: every
   address inside the span is mapped, so a tenant that walks past its boundary reads and writes
   another tenant's live data and raises nothing at all. It surfaces as a wrong *number* many
   layers later, never as a fault.

   The rules, each of which has been violated in production at least once:

   - **The tier may not be placed above `weight_floor`** — `region_pool::tier_fits`.
   - **`weight_floor` may not move below a standing tier's top.** The floor guard refused a
     floor that cut live KV regions and said nothing about the tier, so the weight side could
     grow *down* onto ground a placed tier was already standing on. `tier_fits` had approved
     that tier against the floor as it stood a moment earlier; lowering the floor underneath it
     retroactively put its top inside the weight zone, and the `wave-ffn` span at the tier's
     top wrote activations over resident expert slots. Measured: tier topping out at
     `0x51fbc00000` against a floor of `0x51f9c00000` — 32 MiB of tier inside expert ground.
   - **An arena layout may not cross `weight_floor`.** `plan_wave_transient` records a
     forward's plan even when it cannot place a tier for it, so a wider forward arriving behind
     a live one leaves its plan standing over the previous, narrower tier. The layout then
     walks the wider plan from a base chosen for a smaller purchase.
   - **A raw device address captured from one tenant is invalidated by any boundary move.**
     The MoE dispatch tables cache one slot address per expert on the reasoning that an
     all-resident cache's weights never move. They do: a concession evicts the slots at the
     frontier, and the zone then *grows back* — so capacity and floor read exactly as they did
     at load while the conceded slots hold something else. Compare a monotonic concession
     count, never the geometry.

   **The danger, stated plainly:** a boundary check that consults live occupancy
   (`region_stats().transient_bytes`, a mid-wave snapshot) instead of the reservation, or that
   compares two figures derived from the same array, passes while the invariant is broken. When
   a symptom looks like bad arithmetic — NaN in a GEMM, an implausible magnitude — but the
   operands and weights are individually finite, suspect the partition before the kernel.
   `candle::readonly_regions` (behind `tensor-assert`) exists to catch exactly this: declare a
   tenant's ground immutable and the guard names the writer at the moment of the write, instead
   of leaving a wrong number to be found downstream.

---

## The `tensor-assert` Harness

Everything below is behind the **`tensor-assert`** feature and compiles to nothing without it.
Build it with `--features cuda,tensor-assert`; the call sites are `#[cfg]`-ed out, not
branch-predicted away, so feature-off is genuinely zero cost — including the arguments, which is
why no call site may pass a `&format!(...)`.

| Piece | Where | Answers |
|---|---|---|
| `Tensor::assert("name")` / `QTensor::assert` | `candle-core/src/tensor_assert/` | Is this tensor finite? Async — one kernel, no sync, no readback. Stats land in a slot the drain reads later. |
| `check_now` / `check_now_quant` | same | Same question, **synchronously**, for the one site a capture has armed. |
| `on_bad(cb)` | `tensor_assert/callback.rs` | Fires per finding. Compose these: each new callback narrows the previous one's answer. |
| `nan_capture::checkpoint` | `models/nan_capture.rs` | Async locate (free) → armed synchronous capture (fenced, one site) → dump operands → panic. |
| `readonly_regions` | `candle-core/src/readonly_regions.rs` | Names the *writer* at the moment of the write. Lock-free `O(log n)` over non-overlapping ranges; `assert_writable` is plumbed through every FFI write site. |
| `SlotIntegrity` | `models/expert_lre/slot_integrity.rs` | Did resident weights change since load? Three checks — whole grid, rotating shard, single watched slot — that narrow run → ~64 waves → one layer. |

**The method, which is the actual deliverable.** Test, narrow, test, narrow. Each instrument
bounds a window; the next instrument runs inside that window. Do not build a state machine that
walks the narrowing itself — add another `on_bad` callback and let them compound.

**Two dangers, both learned the hard way:**

- **An instrument that fences suppresses the race it hunts.** These faults reproduce at ~10k t/s
  and stop reproducing when the pipeline is drained: a heavily-fenced build ran 71 minutes clean
  while the production build failed in 5. Every `SlotIntegrity` check synchronises the stream, so
  they are budgeted in fences, not in bytes — the shard scan runs **once per sweep**, not once per
  layer, for exactly this reason. Measure throughput after adding instrumentation; if it drops far
  below production, the run proves nothing. The armed-capture design (async locate, synchronous
  capture only at the one site already named) exists to keep that budget.
- **A stale declared region reports a false positive.** `readonly_regions` must be told when ground
  is legitimately released (`release_below`), or a zone that shrinks leaves declarations behind and
  the guard blames an innocent allocation for reusing freed memory.

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

## Sparse-Latent MoE Engine (`candle-transformers/src/models/latent_moe/`)

The DeepSeek-V4-Flash inference stack. It is split so that **no machinery names a
model version**: `latent_moe/` is the architecture *family* — layers, paged/batched
kernel path, provenance gallery, wave engine — and each concrete model is one file
beside it supplying only what is genuinely its own.

| Where | What |
|-------|------|
| `models/deepseek4.rs` | **The model.** `impl Arch for DeepSeekV4` — config defaults, GGUF metadata keys, tensor names, latent geometry, and the `dflash` drafter arch. Its tests assert the released checkpoint's identity against the real GGUF. |
| `latent_moe/arch.rs` | The `Arch` trait + the `Weight` / `Global` / `Meta` enums naming every tensor and hyperparameter the engine asks for. Exhaustive matches, so a new engine tensor breaks every model at compile time. Also `test_arch`, a synthetic architecture with deliberately *unlike* naming that the engine's own tests run against. |
| `latent_moe/geometry.rs` | `LatentGeometry` — `(head_dim, rope_dim, n_bands)` plus the divisibility rules the kernel tiling depends on. `SUPPORTED` lists the geometries the kernels are built for. |
| `latent_moe/config.rs` | `Config` (carries its `&'static dyn Arch`), `LayerKind`. |
| `latent_moe/loader.rs` | GGUF → weights. Names every tensor through the arch. |
| `latent_moe/{paged,gallery,wave,engine}.rs` | Kernel wrappers, provenance corpus, wave batching, resident model. Model-agnostic. |

**Adding a model in this family** is a sibling of `deepseek4.rs`: a unit struct
implementing `Arch`. If it changes the *latent geometry*, `geometry::SUPPORTED`
documents the three extra kernel-side steps. The kernels are templates over
`<HEAD_DIM, ROPE_DIM, NPAL>`; `paged_latent_api_bf16.cu` pins the live triple, and
`paged::assert_kernel_geometry` refuses a host/kernel mismatch at load — a
divergence there is wrong attention, not a fault.

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

### Debugging a faulting kernel — `kernel-lineinfo`

The kernels build **without** `--generate-line-info`, so a device-side fault
gives you an address and no source location. When you need the file and line —
an illegal access whose origin is not obvious, a `compute-sanitizer` run — build
with the feature for that session and drop it again afterwards:

```bash
cargo test -p candle-transformers --features cuda,kernel-lineinfo <test> -- --nocapture
```

**Do not leave it on.** It costs nothing at runtime and two thirds of the build
on disk: a measured cubin holds 175 KB of `.text` SASS against 592 KB of debug
sections, `.nv_debug_ptx_txt` (the embedded PTX source text) being 490 KB of
that. Those archives are statically linked into *every* CUDA test binary, and
cargo keeps every generation of every binary it has ever produced.

Which is the other half of the same story: `target/` grows tens of GB per build
generation and cargo has no garbage collector. `cargo prune` (`target-prune`)
sweeps superseded generations — it keeps the two newest of each artifact, so
alternating feature sets do not thrash. `cargo prune -- --dry-run` reports first.

---

## Platform Notes

- **Windows + CUDA**: `cuda.dll`, `cublas.dll`, `curand.dll` must be on PATH.
- **WSL**: Never load models from `/mnt/c` — I/O is extremely slow; copy to native Linux FS.
- **Flash attention**: requires `git submodule update --init` and `--features flash-attn`.
