//! Browser demo: BLIP image captioning compiled to WebAssembly.
//!
//! Crate root for the Salesforce BLIP image-captioning demo (float
//! `candle_transformers::models::blip` or GGUF-quantized
//! `quantized_blip`). Provides the `console_log!`/`console.log` bridge and
//! the `token_output_stream` submodule (`TokenOutputStream`, incremental
//! tokenizer decoding for streamed caption tokens). The `#[wasm_bindgen]`
//! surface lives in `src/bin/m.rs`: `Model::load(weights, tokenizer, config,
//! quantized)` then `Model::generate_caption_from_image(image_bytes) ->
//! String`, greedy-sampling caption tokens against vision embeddings
//! computed by BLIP's vision encoder from a browser-supplied image buffer.
use wasm_bindgen::prelude::*;
pub mod token_output_stream;

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
