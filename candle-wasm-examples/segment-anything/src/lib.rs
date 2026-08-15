//! Browser demo: Segment Anything (SAM) interactive point-prompt segmentation.
//!
//! This crate root re-exports `Sam` and `IMAGE_SIZE` from
//! `candle_transformers::models::segment_anything::sam` and provides the
//! `console_log!`/`console.log` bridge. The `#[wasm_bindgen]` surface lives
//! in `src/bin/m.rs`: `Model::new(weights, use_tiny)` builds either the
//! ViT-B or the tiny ViT-T SAM backbone; `set_image_embeddings(image_bytes)`
//! runs the vision encoder once per image and caches the result;
//! `mask_for_point(points) -> MaskImage{mask, image}` runs the lightweight
//! mask decoder against normalized `(x, y, is_positive)` click coordinates.
//! Weights are safetensors fetched by the browser and passed in as bytes.
use candle_transformers::models::segment_anything::sam;
use wasm_bindgen::prelude::*;

pub use sam::{Sam, IMAGE_SIZE};

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
