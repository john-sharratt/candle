# candle-examples

Example binaries for candle: ~100 self-contained programs, most demonstrating
a single upstream model end to end (load weights, tokenize/preprocess, run
inference, print or save output). This crate is a workspace member built with
`cargo run --example <name>`; it is not a library other crates depend on,
though `candle-examples::{device, load_image, hub_load_safetensors, ...}`
(see `src/lib.rs`) provides shared helpers used across the examples: device
selection with CPU/CUDA/Metal fallback, image load/resize/save, audio (`wav`,
`bs1770`, `audio`), token streaming (`token_output_stream`), and safetensors
index resolution (`hub_load_safetensors`, `hub_load_local_safetensors`).

## What's here, by category

- **LLMs / text generation** — `llama`, `llama2-c`, `mistral`, `mixtral`, `qwen`, `phi`, `gemma`, `falcon`, `mamba`, `mamba-minimal`, `rwkv`, `olmo`, `yi`, `granite`, `granitemoehybrid`, `stable-lm`, `deepseekv2`, `chatglm`, `glm4`, `starcoder2`, `replit-code`, `codegeex4-9b`, `helium`, `csm`, `orpheus`.
- **Quantized GGUF** — `quantized`, `quantized-gemma`, `quantized-phi`, `quantized-qwen2-instruct`, `quantized-qwen3`, `quantized-t5`.
- **Embeddings / encoders** — `bert`, `bert_single_file_binary` (self-contained embedded-weights binary, see `bert_single_file_binary_builder/`), `jina-bert`, `debertav2`, `distilbert`, `modernbert`, `xlm-roberta`, `gte-qwen`, `nvembed_v2`, `splade`, `stella-en-v5`.
- **Vision** — `resnet`, `vgg`, `convmixer`, `convnext`, `efficientnet`, `efficientvit`, `mobilenetv4`, `mobileone`, `mobileclip`, `repvgg`, `fastvit`, `hiera`, `dinov2`, `dinov2reg4`, `eva2`, `vit`, `beit`, `yolo-v3`, `yolo-v8`, `segformer`, `segment-anything`, `depth_anything_v2`, `trocr`.
- **Multimodal** — `clip`, `siglip`, `chinese_clip`, `blip`, `llava`, `moondream`, `paligemma`, `pixtral`, `colpali`.
- **Diffusion / generative image** — `stable-diffusion`, `stable-diffusion-3`, `wuerstchen`, `flux`.
- **Audio** — `whisper`, `whisper-microphone`, `encodec`, `mimi`, `snac`, `musicgen`, `metavoice`, `parler-tts`, `voxtral`, `silero-vad`.
- **Training** — `mnist-training`, `llama2-c` (also trains a small model).
- **ONNX** — `onnx`, `onnx-llm`, `onnx_basics.rs`.
- **Misc** — `custom-ops` (writing a custom CUDA/CPU op), `reinforcement-learning` (PyO3 gym bridge), `yolo-v3`/`yolo-v8` (object detection).

Most of the above are inherited from upstream Candle and each demonstrates
one model in isolation — they are not this fork's production path. This
fork's production inference engine is `zend` (the daemon) on top of
`candle-conversation` / `candle-transformers::batched_inference`.

A handful of examples are specific to this fork's engine work rather than
upstream: `llama-parallel-batching`, `qwen2-parallel-batching`,
`qwen3-parallel-batching`, and `llama_multiprocess` exercise the batched
multi-session inference path; `decode_ab` is a correctness/throughput harness
for the paged-decode INT8 kernel; `ruler-eval` and `perplexity-eval` run
long-context (RULER) and WikiText-2 perplexity evaluation across the C0–C9 KV
compression levels for the O(1)-error paper; `test_qwen3_rope_real.rs` checks
Qwen3 RoPE behaviour at long context.

## Running an example

```bash
cargo run --example <name> --release --features cuda -- <args>
```

Real feature names, from `Cargo.toml`: `cuda`, `cudnn`, `metal`, `accelerate`,
`mkl`, `flash-attn`, `nccl`, `onnx`, `microphone`, `encodec`, `mimi`, `snac`,
`depth_anything_v2`, `tekken`, `bert-single-file-binary-builder`. Several
examples declare `required-features` in `Cargo.toml` (e.g. `whisper` needs
`symphonia`, `onnx*` need `onnx`, `mnist-training`/`llama2-c` need
`candle-datasets`, `llama_multiprocess` needs `cuda,nccl,flash-attn`) and will
refuse to build without them. Pass `--cpu` (most examples accept it) to force
CPU execution; otherwise `candle_examples::device()` picks CUDA, then Metal,
then falls back to CPU.
