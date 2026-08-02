# candle-kernels

AOT-compiled CUDA kernels for the paged, quantized inference stack — FFI
bindings from `candle-core`'s CUDA backend into ~720 precompiled `.cu` kernels.

## What it does

This crate has almost no logic of its own: it is a build pipeline
(`build.rs` + `build_utils.rs`) that compiles every `.cu` file under `src/`
with `nvcc`, links them into a handful of static archives, and exposes typed
Rust FFI wrappers (`src/lib.rs` and the per-subdirectory `api.rs` files) that
`candle-core/src/cuda_backend/` calls with `unsafe`. Kernels are grouped into
**archive groups** compiled with matching flags — `simple`, one archive per
quantized dtype (from `QUANTIZED_KERNELS`), `sampling`, `provenance`,
`paged_prefill`, `paged_decode`, `paged_glue` — each linked into `lib<name>.a`
and gzip-compressed into `precompiled/` for reuse across builds.

`CHUNK_SIZE = 32` (`src/lib.rs`, mirrored by `#define CHUNK_SIZE 32` in
`src/blocks.cuh`) is the shared Rust/CUDA constant: every paged-attention
kernel and every quantized block format operates on 32-token/32-element
blocks, so this value must never drift between the two languages.

## Build pipeline (`build.rs` / `build_utils.rs`)

1. Every `.cu`/`.cuh` under `src/` is registered for `cargo:rerun-if-changed`.
2. Each archive group's aggregate hash is computed from a **SHA-256 of every
   kernel's source plus its transitively `#include`d headers plus its
   compile flags** (line-ending normalized). If the aggregate hash matches
   `precompiled/lib<name>.a.sha256`, the group is unpacked from
   `precompiled/lib<name>.a.gz` and nvcc never runs.
3. For groups that changed, individual kernels are checked against a
   `staged/` `.o` cache keyed by the same per-kernel hash — so editing one
   `.cu` file recompiles only that kernel, not its whole archive group.
4. Dirty kernels compile in parallel (bounded to `min(cores, 16)` nvcc jobs)
   targeting `sm_89` (Ada) native SASS plus `compute_120`/`sm_120`
   (Blackwell) PTX-and-SASS as a forward-compat fallback.
5. Archives are relinked (`ar`/`lib.exe`), recompressed into `precompiled/`,
   and the new aggregate hash is saved.
6. Cargo link directives are emitted for every archive plus `cudart` (and
   `stdc++` off MSVC).

Run `make clean-ptx` to wipe `precompiled/`/`staged/` and force a full
recompile — needed after a compiler/toolkit upgrade or if the cache is
suspected stale.

## Key subdirectories

| Path | Role |
|------|------|
| `src/simple/` | Elementwise/binary/ternary ops, indexing, sort, reduce, casts, arena compaction, KV migration (`kv_migrate.cu`), MoE routing (`moe_bucketize.cu`, `moe_scatter.cu`), Fletcher-32 checksums |
| `src/quantized/`, `src/quantized/impl/` | Quantized batched matmul dispatcher + per-format GEMM instantiations (Q2_K…Q8_K, AWQ) × {F16, BF16, F32} output |
| `src/quantize/` | GPU-side KV quantize kernels — the on-GPU counterpart to `candle-nn`'s `CompressionPolicy` format conversion |
| `src/dequant/`, `src/convert/` | Per-format dequantize/block-convert device headers shared across kernels |
| `src/provenance/` | Binary Directional Provenance scan: `bdp_scan.cu` (scalar), `bdp_bmma.cu` (b1 tensor-core, sm_75–sm_89), `bdp_imma.cu` (INT8 tensor-core, sm_80+ incl. Blackwell), sharing `bdp_vote.cuh` |
| `src/paged-decode/` | Paged-attention decode kernels, dispatched per dtype (fp16/bf16) and head dim |
| `src/paged-prefill/` | INT8 prefix-attention prefill kernels |
| `src/paged-glue/` | Reprojection "glue" forward — a decode-derivative kernel for the small structural-token islands assembled between sealed chunks at reproject time |
| `src/sampling/` | Batched fused sampling kernel (temperature/top-k/top-p/repetition penalty etc. in one launch) |
| `src/arena_table.cuh`, `src/blocks.cuh` | Shared device headers: the arena-table row layout (`ArenaTableEntry`, format-tag encoding) and every K/V block struct (`BlockQ4_0`, …), matching their Rust counterparts byte-for-byte via `static_assert` |

## Adding a kernel

1. Add the `.cu` file under the appropriate `src/<subdir>/`.
2. Register it in the relevant `const [...]_KERNELS` array (and, if it needs
   a new archive group, add one) in `candle-kernels/build_utils.rs` — this is
   what feeds the SHA-256 cache and the nvcc compile/link step.
3. Add an FFI binding: an `extern "C"` launcher wrapper in the subdirectory's
   `api.rs`/`api.cu` pair, exposed as `pub mod` in `src/lib.rs`.
4. Call it from `candle-core/src/cuda_backend/` via `unsafe`, matching the
   `candle_kernels::<module>::<fn>` signature.

## How it is used

Only `candle-core` depends on this crate, gated behind the `cuda` feature
(`candle-core/Cargo.toml`: `cuda = [..., "dep:candle-kernels", ...]`).
Everything above `candle-core` (`candle-nn`'s KV cache, `candle-transformers`,
`candle-conversation`) reaches these kernels indirectly through `Tensor`/
`QTensor` ops and the CUDA backend's paged-attention/quantized-matmul entry
points — nothing outside `candle-core` calls into `candle-kernels` directly.

## Related docs

- `docs/glue_prefill_kernel.md` — the batched glue-prefill kernel design
  (`src/paged-glue/`, `paged-prefill`'s `GAP_FILL` specialization)
- `docs/paged_gallery_arena.md` — the provenance scan's resident VRAM gallery
  and the `bdp_bmma.cu`/`bdp_imma.cu` tensor-core backends (§14)
- `docs/gpu_native_moe_dispatch.md` — GPU-native MoE expert dispatch built on
  `src/simple/moe_bucketize.cu` and `moe_scatter.cu`
