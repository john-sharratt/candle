//! Browser demo: Microsoft Phi-1.5 / Phi-2 causal LM compiled to WebAssembly.
//!
//! This crate root only provides the `console_log!`/`console.log` bridge;
//! the model logic lives in `src/bin/m.rs`, which loads either the float
//! `MixFormerSequentialForCausalLM` or the GGUF-quantized `QMixFormer`
//! behind a `#[wasm_bindgen]` `Model` (selecting phi-2's `new_v2`
//! constructor when the config's `_name_or_path` says so). Surface:
//! `Model::load(weights, tokenizer, config, quantized)`,
//! `init_with_prompt(prompt, temp, top_p, repeat_penalty, repeat_last_n,
//! seed) -> String`, and `next_token() -> String` for autoregressive
//! decoding one token at a time. Weight bytes are fetched by JS and passed in.
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
