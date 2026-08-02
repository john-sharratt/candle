# candle-metal-kernels

Metal compute-shader kernels for `candle-core`'s Apple GPU backend.

## What it does

This crate is the Metal counterpart to `candle-kernels`: instead of AOT-compiled
CUDA/PTX, it ships MSL (`.metal`) shader source embedded via `include_str!` and
compiles it to a `MTLLibrary`/`MTLComputePipelineState` at runtime through
`objc2-metal`. It covers the same op surface `candle-core`'s CPU/CUDA backends
provide — affine, binary/unary/ternary elementwise ops, casts, reductions,
sorting, fills, random generation, indexing/scatter, convolution, quantized
matmul, an MLX-derived GEMM, and scaled-dot-product attention — as well as a
`Kernels` cache that loads each Metal library once and memoizes compiled
pipelines by `(name, constants)`.

This fork's production inference path (batched paged attention, adaptive KV
quantization, provenance retrieval) targets **CUDA only**; this crate exists
so `candle-core`'s `metal_backend` and the upstream model zoo in
`candle-transformers` keep working on macOS, not because Metal is a target
platform for the unbounded-context engine.

## Key modules / layout

| Path | Role |
|------|------|
| `src/lib.rs` | Crate root: re-exports, `DType` (the Metal-side element type enum), `RESOURCE_OPTIONS`. |
| `src/kernel.rs` | `Kernels` — the library/pipeline cache (`load_library`, `load_pipeline[_with_constants]`). |
| `src/source.rs` | `Source` enum mapping each kernel family to its embedded `.metal` source string. |
| `src/kernels/` | One Rust module per kernel family (`affine`, `binary`, `cast`, `convolution`, `fill`, `indexing`, `mlx_gemm`, `multinomial`, `quantized`, `random`, `reduce`, `sdpa`, `sort`, `ternary`, `unary`, `*_at_indices`) — each builds encoder arguments and dispatches its pipeline. |
| `src/metal_src/*.metal` | The actual MSL shader source, one file per kernel family. |
| `src/metal/` | A thin wrapper layer over `objc2-metal` (`Device`, `Library`, `ComputePipeline`, `CommandBuffer`, `Encoder`, `Buffer`) providing a more ergonomic Rust API over the raw Objective-C bindings. |
| `src/err.rs`, `src/utils.rs` | `MetalKernelError`, and dispatch helpers (`linear_split`, `get_block_dims`, `EncoderParam`/`EncoderProvider`). |

## How it is used

`candle-core` depends on this crate behind its `metal` feature
(`candle-core/Cargo.toml`: `metal = ["dep:objc2-metal", "dep:objc2-foundation", "dep:candle-metal-kernels", "dep:ug-metal"]`)
and calls into it from `candle-core/src/metal_backend/`. Callers construct one
`Kernels` per `Device`, then call the `call_*` functions re-exported from
`kernels::*` (e.g. `call_binary_contiguous`, `call_sdpa_full`,
`call_quantized_matmul_mm_t`) with a command buffer/encoder and buffer
offsets. No CUDA/`candle-kernels` code paths are shared with this crate — it
is entirely independent of `candle-kernels`' AOT-CUDA pipeline.
