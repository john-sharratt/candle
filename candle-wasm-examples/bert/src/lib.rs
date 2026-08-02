//! Browser demo: BERT sentence embeddings compiled to WebAssembly.
//!
//! This crate root re-exports `BertModel`/`Config`/`DTYPE` from
//! `candle_transformers::models::bert` and the `tokenizers` types the wasm
//! binary needs, plus a `console_log!` macro bound to the browser's
//! `console.log`. The actual `#[wasm_bindgen]` surface — `Model::load(weights,
//! tokenizer, config)` and `Model::get_embeddings(params) -> Embeddings`
//! (mean-pooled, optionally L2-normalized) — lives in `src/bin/m.rs`; weight,
//! tokenizer and config bytes are fetched by JS and passed in, none are
//! embedded in the wasm binary. Consumed from `bertWorker.js` in a WebWorker.
use candle_transformers::models::bert;
use wasm_bindgen::prelude::*;

pub use bert::{BertModel, Config, DTYPE};
pub use tokenizers::{PaddingParams, Tokenizer};

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

#[macro_export]
macro_rules! console_log {
    // Note that this is using the `log` function imported above during
    // `bare_bones`
    ($($t:tt)*) => ($crate::log(&format_args!($($t)*).to_string()))
}
