# candle-nn

Neural-network building blocks on top of `candle-core`, and the home of this
fork's paged, adaptively-quantized KV cache — the most actively developed
crate in the repository.

## What it does

`candle-nn` has two halves. The first is an ordinary layer library inherited
from upstream Candle: `Linear`, `LayerNorm`/`RmsNorm`, `Embedding`, `Conv1d`/
`Conv2d` (+ transpose variants), `BatchNorm`, `GroupNorm`, RNNs (`GRU`/`LSTM`),
activations, loss functions, the `AdamW`/`SGD` optimizers, and the
`VarBuilder`/`VarMap` weight-loading machinery. These are the standard
components any transformer model in `candle-transformers` is assembled from.

Around the layer library sit a few smaller building blocks: `ops.rs`
(`Dropout` and misc tensor ops), `sampling.rs` (Gumbel-Softmax sampling),
`encoding.rs` (one-hot/cold encoding), `cpu_flash_attention.rs` (a CPU
reference flash-attention implementation), `kv_caches.rs` (`KvCaches` — a
multi-layer container pairing per-layer `KvCache`s with a shared causal-mask
cache), and `sequence_context.rs` (`SequenceContext` — bundles a sequence's KV
cache reference, position offset, and input tokens for continuous batching).

The second half — `src/kv_cache/` — is this fork's real work: a paged,
per-block-quantized, three-tier KV cache built to support unbounded context at
O(1) numerical error per decode step. Every attention layer's K/V state is
owned by one of several cache types depending on how it needs to move and
compress; the chunked (paged) cache is the one the production decode/prefill
path uses, because it is the only one that supports Arc-based prefix sharing,
per-block quantization, and hot/warm/cold tiering. The other cache types exist
for narrower needs: `Cache`/`KvCache` are the dense contiguous baseline used
in tests and simple single-sequence models; `RotatingKvCache` implements a
fixed-size sliding window (e.g. local-attention layers); `ScatteredKvCache`
handles sparse, batch-masked writes.

## Key modules / layout

| Path | Role |
|------|------|
| `src/kv_cache/mod.rs` | Public API: `KvFormat`, `QuantFormat` (21 quantized formats), `PagedKvArenas` trait |
| `src/kv_cache/cache.rs` | `Cache`, `KvCache` — contiguous dense caches; also the internal `ChunkedCache` wrapper |
| `src/kv_cache/rotating.rs` | `RotatingCache`/`RotatingKvCache` (fixed-size sliding window), `ScatteredKvCache` (sparse, batch-masked) |
| `src/kv_cache/arena_table.rs` | `ArenaTable`, `PerHeadTable`, `ArenaLocation`, `ArenaFormatTag` — the GPU-indexable metadata tables kernels read directly |
| `src/kv_cache/chunked/backing.rs` | `ChunkedKvBacking` — the paged cache's core shared state (`BackingInner`) |
| `src/kv_cache/chunked/arena.rs` | `Arena`, `ArenaKey`, `ArenaStorage`, `StoragePolicy` — one arena per (format, location) pair |
| `src/kv_cache/chunked/gid_pool.rs` | `ChunkGidPool` — lock-free, RAII, refcounted global chunk-slot allocator |
| `src/kv_cache/chunked/head_gids.rs` | `HeadGids` — per-block, per-(head, palette, K/V) chunk GID collection |
| `src/kv_cache/chunked/compress.rs` | Quantize-on-evict: per-`(chunk, head, palette)` format selection + conversion kernels at the hot→warm boundary |
| `src/kv_cache/chunked/compression_policy.rs` | `CompressionPolicy`, the C0–C10 candidate-format ladders, production K/V error thresholds |
| `src/kv_cache/chunked/sequence_ops.rs` | Sequence alloc/free, prefix sharing (COW), forking (beam search / speculative decode) |
| `src/kv_cache/chunked/io.rs` | `read_contiguous`/`write_contiguous` — dequantize-on-read for testing/CPU fallback |
| `src/kv_cache/chunked/gpu_chunks.rs` | Per-sequence pinned-host + device slot-state buffer feeding the paged kernels' `get_slice` |
| `src/kv_cache/chunked/types.rs` | `ChunkMeta`, `SealedChunk`/`SealedSequence`, `ChunkWindow`, `CHUNK_SIZE = 32` |
| `src/kv_cache/chunked/migrate.rs`, `meta_pool.rs` | `kv_migrate` (VRAM↔RAM tier migration kernel wrapper), `MetaGid`/`MetaPool` (device-resident per-chunk head metadata) |
| `src/kv_cache/chunked/sampled_selection/` | The CPU/GPU format-selection kernels and their calibration/benchmark tests |

## Key types & entry points

- **`KvFormat`** — `Float(DType)` or `Quantized(QuantFormat)`. All quantized
  blocks are 32 elements (`CHUNK_SIZE = 32`, shared with `candle-kernels`).
  `QuantFormat` spans 21 formats from `R16` (near-lossless, 128 B/32-elem) down
  to `Q0`/`Q0_V`/`Q0_X` (~0.25–0.5 bits/element).
- **`CompressionPolicy`** — carried by the conversation/session layer, not the
  allocator. Its presence *is* the "adaptive compression on" toggle; `None`
  means uniform, unquantized storage. `compression_level` selects one of 11
  production candidate ladders (C0 near-lossless → C10 maximum compression,
  `PRODUCTION_K_CANDIDATE_FORMATS`/`PRODUCTION_V_CANDIDATE_FORMATS`). At seal
  time the selection kernel evaluates every candidate in the level's ladder
  against a per-block cosine-distance threshold and picks the smallest format
  that passes. K and V have **separate** threshold tables
  (`PRODUCTION_K_QREL_HIGH/LOW_THRESHOLDS`, `PRODUCTION_V_QREL_HIGH/LOW_THRESHOLDS`)
  because K is sensitive by channel and V by token. `Q4_KS`/`Q8_KS` carry a
  dedicated sub-block fine scale for attention-sink protection: positions 0–3
  of every sequence get finer quantization so their large attention-sink
  magnitudes don't inflate the block's shared scale for the other 28 tokens.
- **`ChunkedKvBacking`** — the paged cache. Chunks (32 tokens each) live in
  format-specific `Arena`s; `ChunkGidPool` hands out RAII `ChunkGid`s that
  auto-return to a lock-free free list on drop; `HeadGids` bundles the
  `N_PALETTE (4) × 2 (K/V) × n_kv_head` GIDs one logical block owns. Sequences
  share prefix chunks by cloning `Arc`s (copy-on-write), so forking a
  conversation for a new turn or a speculative branch is O(shared blocks), not
  O(context length).
- **`PagedKvArenas`** trait — the abstraction attention kernels use to read K/V
  arenas (float or quantized) without knowing the concrete cache type.

## Three-tier residency

Chunks migrate GPU (hot) → RAM (warm, pageable CPU arenas) → NVMe (cold, the
append-only redo log at `.substrate/substrate.log`, owned by
`candle-conversation`'s persistence layer). Hot→warm quantizes in place via
`quantize_sealed_in_place` when a `CompressionPolicy` is active, then does a
format-preserving device-to-host copy; warm→hot elevates back on demand. Two
divergences from the target design remain (see `docs/kv_tier_migration.md`):
warm residency is pageable rather than pinned, and hot→warm currently runs on
the primary CUDA stream rather than a dedicated overlap stream.

## Testing

Per the repo-wide TDD convention, `chunked/tests/` and `sampled_selection/tests/`
hold unit tests alongside every submodule they exercise (arena allocation,
GID pool refcounting, sequence forking, compression selection, the CPU/GPU
selection kernels' bit-exact agreement). Quantization/serialization tests
assert against raw expected bytes rather than error-tolerance thresholds —
`chunked/fletcher_golden.rs` and `sampled_selection/tests/test_data.rs` hold
the golden fixtures. GPU-dependent tests are gated behind `#[cfg(feature =
"cuda")]` and `candle::test_device!`, which serializes per-device execution
via `serial_test`.

## How it is used

`candle-transformers` builds every model's attention layers against these
cache types; `candle-conversation` (substrate, projection, scheduler) owns the
`CompressionPolicy` and drives sequence lifecycle (alloc, fork, seal, migrate).
Everything under `chunked/` and the CUDA-only re-exports in `kv_cache/mod.rs`
require the `cuda` feature — the CPU float caches (`Cache`, `KvCache`,
`RotatingKvCache`, `ScatteredKvCache`) build without it. Other relevant
features: `cudnn` (adds cuDNN-backed ops via `candle`), `metal`, `accelerate`,
`mkl` (BLAS backends for the plain layer library).

## Related docs

- `docs/kv_tier_migration.md` — three-tier storage design, current build status
- `docs/coresident_kv_metadata.md` — device-resident per-chunk head metadata
- `docs/attention_provenance.md` — how the provenance scan selects which
  chunks (across all tiers) participate in a given attention step
