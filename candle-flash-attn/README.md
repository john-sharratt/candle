# candle-flash-attn

CUDA binding crate for FlashAttention v2, exposed to `candle-core` `Tensor`s as a custom op.

## What it does

`src/lib.rs` defines `FlashAttn`/`FlashAttnVarLen` as `candle::CustomOp3` impls
whose `cuda_fwd` calls `ffi::run_mha` (declared in `src/ffi.rs`) — a C FFI
binding into the Dao-AILab flash-attention v2 kernel sources vendored under
`kernels/` (one specialized `.cu` per head-dim × dtype × causal combination,
`flash_fwd_hdim{32..256}_{fp16,bf16}[_causal]_sm80.cu`, sharing
`flash_fwd_kernel.h`/`flash_fwd_launch_template.h`). The public functions —
`flash_attn`, `flash_attn_windowed`, `flash_attn_alibi[_windowed[_softcap]]`,
and the `_varlen` variants for ragged/packed batches — implement
`softmax(Q @ Kᵀ · scale) @ V` with causal masking, sliding-window limits,
ALiBi slopes, and Gemma-style softcap, for `f16`/`bf16` inputs only (no CPU
fallback — `cpu_fwd` unconditionally errors). Head dimension must be ≤ 256 and
a multiple of 8; GQA/MQA is supported (`k`/`v` head count must divide `q` head
count).

`build.rs` invokes `nvcc` directly (not `candle-kernels`' build path) to
compile the kernel list into `libflashattention.a`, using CUTLASS (vendored as
a git submodule at `candle-flash-attn/cutlass`) for its GEMM primitives. It
SHA256-hashes each kernel plus its headers and build flags to skip
recompilation when nothing changed (`CANDLE_FLASH_ATTN_BUILD_DIR` can point
the cache at a persistent directory). Even cached, a cold build compiles ~30
kernel specializations and is **slow** — expect it to dominate a clean build's
wall time.

## Requirements

- `git submodule update --init` (pulls in `candle-flash-attn/cutlass`) — the build fails without it.
- Build with `--features flash-attn` (from `candle-transformers`) or `cuda` + this crate directly; it has no `default` features and always requires CUDA (`candle-core` is pulled in with `features = ["cuda"]`).
- `nvcc` on `PATH`; Windows builds link via `lib.exe`, Unix via `ar`.

## Where it's used from

`candle-transformers`'s `flash-attn` feature (`flash-attn = ["cuda", "dep:candle-flash-attn"]`)
gates `candle_flash_attn::flash_attn(...)` calls scattered through the
**upstream, non-batched** model forward implementations — `qwen3.rs`,
`llama.rs`, `mistral.rs`, `mixtral.rs`, `gemma.rs`/`gemma2.rs`/`gemma3.rs`,
`granite.rs`, `granitemoehybrid.rs`, `helium.rs`, `stable_lm.rs`,
`voxtral_llama.rs`, and the diffusion/audio attention blocks in
`stable_diffusion/`, `mmdit/`, `mimi/`, `wuerstchen/` — each behind a
model-level `use_flash_attn: bool` flag. It also backs
`batched_layer.rs::prefill_attention_simple`, a per-sequence fallback attention
path documented as used "for quantized KV cache mode where paged CUDA kernels
aren't available."

**It is not part of this fork's paged/provenance production attention path.**
The batched engine's real prefill route is the custom paged-attention CUDA
kernels in `candle-kernels` (`paged_prefill_flat`/`paged_prefill_batched` in
`candle-transformers/src/models/prefill_utils.rs`), which implement INT8/BF16
attention directly over the chunked/paged/quantized KV arena — FlashAttention
has no notion of paging or per-block adaptive quantization. `candle-flash-attn`
remains useful as a fast, well-tested dense-attention primitive for the
standalone model zoo and as a correctness fallback, not as the engine's
critical path.
