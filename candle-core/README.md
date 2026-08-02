# candle-core

The tensor foundation of the stack: `Tensor`, `Device`, `DType`, shape/layout,
op dispatch, autograd, and the feature-gated CPU/CUDA/Metal backends everything
above it is built on.

## What it does

`candle-core` provides the N-dimensional `Tensor` type and the machinery to
compute with it: elementwise/binary/reduce ops, matmul, convolution,
broadcasting, indexing, and reverse-mode autodiff (`backprop.rs`, `Var`). A
`Tensor` holds an `Arc<RwLock<Storage>>` plus a `Layout` (shape + strides +
offset) and a `BackpropOp` recording how it was produced, so gradients can
flow back through arbitrary op graphs.

`Storage` is an enum over three backends — `CpuStorage`, `CudaStorage`,
`MetalStorage` — selected by the `Device` a tensor lives on (`Device::Cpu`,
`Device::Cuda(CudaDevice)`, `Device::Metal(MetalDevice)`). Each backend
implements the same `BackendDevice`/`BackendStorage` traits (`src/backend.rs`),
so op dispatch in `tensor.rs` is backend-agnostic; the actual numeric work
lives in `src/cpu_backend/`, `src/cuda_backend/`, `src/metal_backend/`.

This fork also carries `src/quantized/` (GGML/GGUF-format quantized tensors —
`QTensor`, `GgmlDType`, the CUDA quantized-matmul and KV-quantization kernels
entry points), `src/vram.rs` / `src/gpu_memory.rs` / `src/gpu_poison.rs` (VRAM
accounting and the governor's allocation guards), and `src/fletcher.rs`
(Fletcher-32 checksums used to validate tier-migrated KV data).

## Key modules / layout

| Path | Role |
|------|------|
| `src/tensor.rs` | `Tensor`, `TensorId` — the core type; op dispatch to the active backend |
| `src/device.rs` | `Device`, `DeviceLocation`, `NdArray` |
| `src/dtype.rs` | `DType`, `WithDType`, `FloatDType`/`IntDType` |
| `src/layout.rs`, `src/shape.rs`, `src/strided_index.rs` | Shape/stride bookkeeping, broadcasting, iteration order |
| `src/storage.rs` | `Storage` enum unifying the three backends |
| `src/backend.rs` | `BackendDevice`/`BackendStorage` traits every backend implements |
| `src/backprop.rs`, `src/variable.rs` | Reverse-mode autodiff graph walk; `Var` (a tensor that tracks gradients) |
| `src/op.rs`, `src/custom_op.rs` | Op enums (`UnaryOp`, `BinaryOp`, `ReduceOp`, `CmpOp`) and the `CustomOp1/2/3` extension points |
| `src/cpu_backend/` | Scalar + Rayon-parallel CPU op implementations (`CpuStorage` variants: U8/U32/I64/BF16/F16/F32/F64/F8E4M3) |
| `src/cuda_backend/` (feature `cuda`) | CUDA backend via `cudarc`; FFI entry points into `candle-kernels`; kernel-launch breadcrumb ring for async-error attribution |
| `src/metal_backend/` (feature `metal`) | Metal backend via `candle-metal-kernels` |
| `src/quantized/` | `QTensor`, GGML/GGUF loaders (`ggml_file.rs`, `gguf_file.rs`), `k_quants.rs` (block format structs), per-arch dequant fallbacks (`avx.rs`, `neon.rs`, `simd128.rs`), `cuda.rs` (quantized matmul, KV migration, pinned staging) |
| `src/safetensors.rs`, `src/npy.rs`, `src/pickle.rs` | Checkpoint format loaders |
| `src/vram.rs`, `src/gpu_memory.rs`, `src/gpu_poison.rs` | VRAM budget accounting and OOM-guard primitives used by the KV cache's VRAM governor |
| `src/fletcher.rs` | Fletcher-32 checksum (golden-record validation for KV tier migration) |
| `src/error.rs` | `Error`, `Result`, `Context` — the crate-wide error type |

## Key types & entry points

- **`Tensor`** — the value type everything operates on; `Tensor::arange`,
  `::zeros`, `::from_slice`/`::new`, `.reshape`, `.matmul`, `.to_dtype`,
  `.to_device`, `.backward()`.
- **`Device`** — `Cpu`, `Cuda(CudaDevice)`, `Metal(MetalDevice)`; determines
  which backend a tensor's storage uses.
- **`DType`** — `U8`, `U32`, `I64`, `BF16`, `F16`, `F32`, `F64`, `F8E4M3`.
- **`QTensor`** / `GgmlDType` (`src/quantized/`) — quantized weight and KV
  storage; the GGML block types (`BlockQ4_0`, `BlockQ8_0`, …) that
  `candle-nn`'s `QuantFormat` maps onto are defined in `k_quants.rs`.
- **`Module`/`ModuleT`** traits (re-exported by `candle-nn`) — the `forward`
  contract every layer implements; defined in `candle-nn`, not here, but
  `candle-core` is what their tensors are built from.

## How it is used

Every other crate in the stack (`candle-nn`, `candle-transformers`,
`candle-conversation`) depends on `candle-core` for `Tensor`/`Device`/`DType`.
`candle-kernels` provides the compiled CUDA kernels that `src/cuda_backend/`
calls into via `unsafe` FFI — building with `cuda` pulls in `candle-kernels`
as a dependency (`cuda = ["cudarc", "dep:candle-kernels", "dep:ug-cuda", ...]`
in `Cargo.toml`).

Feature flags: `cuda` (required for essentially all production paths in this
fork — the paged KV cache, quantized inference, provenance scan), `cudnn`
(adds cuDNN via `cudarc/cudnn`, implies `cuda`), `metal` (Apple GPU backend),
`mkl` / `accelerate` (BLAS-accelerated CPU backend on Intel/Apple). Without any
GPU feature, only the CPU backend is available — sufficient for tests but not
for the production inference path, which assumes CUDA.

On Windows, the `windows` crate (DXGI bindings) is pulled in as a
target-specific dependency for the VRAM Governor's DXGI VRAM-usage probe —
compiled only on Windows, used only under `cuda`.

## Examples & benchmarks

`Cargo.toml` declares `cuda_basics` (requires `cuda`) and `metal_basics`
(requires `metal`) example binaries, a `quantized_matmul_benchmark` example
(requires `cuda`), and a `bench_main` Criterion benchmark harness — useful
starting points for exercising a backend directly without going through
`candle-nn`/`candle-transformers`.

## Related docs

- `docs/vram_governor_design.md` — the VRAM accounting/eviction policy built
  on `src/vram.rs` and `src/gpu_memory.rs`
- `CLAUDE.md` — crate stack overview and build commands
