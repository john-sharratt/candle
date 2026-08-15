# bert-single-file-binary-builder

Build-time-only crate (`src/lib.rs` is intentionally empty — only `build.rs`
runs) that fetches the `sentence-transformers/all-MiniLM-L6-v2` model files
(`config.json`, `tokenizer.json`, `model.safetensors`, pinned to commit
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf`) from Hugging Face directly over
`ureq` and writes them into `files/` at compile time. It exists to support
the `bert_single_file_binary` example
(`candle-examples/examples/bert_single_file_binary/`), which
`include_bytes!`/embeds those files so the resulting binary is a
self-contained BERT embedding tool with no runtime download or `hf-hub`
dependency — see the design rationale in
[huggingface/candle#3104](https://github.com/huggingface/candle/pull/3104#issuecomment-3369276760).

`build.rs` skips downloading any file that already exists in `files/`, so
re-runs are cheap once the files are cached locally.

Enabled via the `candle-examples` feature `bert-single-file-binary-builder`
(which pulls this crate in as `dep:bert-single-file-binary-builder`); the
example itself is gated with `required-features =
["bert-single-file-binary-builder"]` in `candle-examples/Cargo.toml`:

```bash
cargo run --example bert_single_file_binary --release --features bert-single-file-binary-builder -- --prompt "hello world"
```

## Limitations

Because the model files must exist on disk before the example's own
`include_bytes!` runs, the model id/revision is hardcoded in `build.rs` and
the files are fetched with a plain HTTP client rather than `hf-hub` — using
`hf-hub`'s cache layout would require navigating hashed snapshot directories
whose paths aren't known until after the download, which is incompatible
with `include_bytes!`'s compile-time path requirement.
