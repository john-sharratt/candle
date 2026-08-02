//! Build-time asset fetcher for the `bert_single_file_binary` example.
//!
//! Has no runtime API of its own — `build.rs` downloads the MiniLM-L6-v2
//! `config.json`, `tokenizer.json`, and `model.safetensors` from the
//! `sentence-transformers/all-MiniLM-L6-v2` HF repo into `files/` at a pinned
//! commit, skipping the fetch if the files already exist. The sibling example
//! embeds those files with `include_bytes!` to produce a single self-contained
//! binary with no HF Hub access at runtime.

// NOTE: this library is intentionally empty as only a build step is needed.
