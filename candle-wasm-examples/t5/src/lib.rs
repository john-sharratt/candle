//! Browser demo: T5 sentence embeddings and seq2seq generation, compiled to
//! WebAssembly.
//!
//! This crate root only provides the `console_log!`/`console.log` bridge;
//! it backs two wasm binaries that share the same `#[wasm_bindgen]` shape.
//! `src/bin/m.rs` loads float safetensors weights into
//! `candle_transformers::models::t5::{T5EncoderModel, T5ForConditionalGeneration}`;
//! `src/bin/m-quantized.rs` loads GGUF weights into the `quantized_t5`
//! equivalents. Both expose `ModelEncoder::decode(sentences) -> embeddings`
//! (mean-pooled) and `ModelConditionalGeneration::decode(prompt, ...) ->
//! generation` (autoregressive text generation with a `LogitsProcessor`).
//! Weight/tokenizer/config bytes are fetched by JS and passed in.
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
