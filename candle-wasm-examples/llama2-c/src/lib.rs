//! Browser demo: karpathy/llama2.c tiny Llama checkpoints, two UI paths.
//!
//! This crate root wires up a pure-Rust Yew UI (`App`, built with Trunk from
//! `src/bin/app.rs`) backed by a `yew_agent` `Worker` (`src/bin/worker.rs`)
//! that owns the `model` submodule's llama2.c `Llama` model and KV cache.
//! A second, JS-driven path (`src/bin/m.rs`) wraps the same `worker::Model`
//! in a `#[wasm_bindgen]` `Model` with `get_seq_len`, `init_with_prompt`,
//! and `next_token`, for use from a plain WebWorker built via
//! `build-lib.sh`. Model/tokenizer bytes come from `wget`-ed
//! `model.bin`/`tokenizer.json` (Trunk build) or are fetched by JS and
//! passed in (WebWorker build).
mod app;
pub mod model;
pub mod worker;
pub use app::App;
pub use worker::Worker;
