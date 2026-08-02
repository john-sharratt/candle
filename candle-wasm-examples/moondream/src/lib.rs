//! Browser demo: Moondream2 vision-language model compiled to WebAssembly.
//!
//! This crate root only provides the `console_log!`/`console.log` bridge;
//! the model logic lives in `src/bin/m.rs`, which wraps float
//! (`candle_transformers::models::moondream::Model`) or GGUF-quantized
//! (`quantized_moondream::Model`) variants behind a `#[wasm_bindgen]`
//! `Model`. Surface: `Model::load(weights, tokenizer, quantized)`,
//! `set_image_embeddings(image_bytes)` to encode a browser-supplied image,
//! then `init_with_image_prompt(input) -> Output{token, token_id}` and
//! `next_token()` to stream a text answer token-by-token. Weight bytes are
//! fetched by JS and passed in.
use wasm_bindgen::prelude::*;

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
