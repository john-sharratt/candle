# tensor-tools

CLI for inspecting and converting tensor checkpoint files (safetensors, NPZ,
GGML, GGUF, PyTorch `.pth`/pickle). Workspace member, single binary crate
(`src/main.rs`), built on `candle::quantized` and `clap`.

## Subcommands

- **`ls <files...> [--format <fmt>] [--verbose] [--metadata-keys] [--metadata-filter <substr>...] [--metadata-values]`**
  — list tensor names with shape/dtype. For GGUF, `--metadata-keys` lists
  metadata keys only (skips tensor listing, useful for huge tokenizer
  arrays), optionally filtered case-insensitively by `--metadata-filter`
  and shown with compact values via `--metadata-values`.
- **`print <file> [names...] [--format <fmt>] [--full] [--line-width <n>]`**
  — print tensor contents; empty `names` prints every tensor. `--full`
  disables truncation (`candle::display::set_print_options_full`);
  `--line-width` sets wrap width. Not supported for `Pickle` format.
- **`quantize <in_files...> --out-file <path> --quantization <q> [--mode llama]`**
  — quantize safetensors (multiple files merged) or a single GGUF file to a
  new GGUF file. `--quantization` accepts `q4_0`, `q4_1`, `q5_0`, `q5_1`,
  `q8_0`, `q8_1`, `q2k`..`q8k`, `f16`, `f32`. `--mode llama` (the only mode)
  quantizes only rank-2 tensors whose name ends `.weight`, forcing
  `output.weight` to `Q6_K` and using safetensors row-count to decide
  per-tensor whether a shape is quantizable (divisible by the format's block
  size).
- **`dequantize <in_file> --out-file <path>`** — read a GGUF file and write
  every tensor, dequantized, to a safetensors file.

`--format` (`safetensors`/`npz`/`ggml`/`gguf`/`pth`) is inferred from the
input file extension when omitted; `.bin` cannot be inferred and requires an
explicit flag.

## Usage

```bash
cargo run -p tensor-tools --release -- ls model.gguf --metadata-keys
cargo run -p tensor-tools --release -- print model.safetensors lm_head.weight --full
cargo run -p tensor-tools --release -- quantize model.safetensors --out-file model-q4k.gguf --quantization q4k
```
