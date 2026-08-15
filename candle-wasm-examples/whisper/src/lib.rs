//! Browser demo: Whisper speech-to-text compiled to WebAssembly.
//!
//! This crate root wires up a pure-Rust Yew UI (`App`, built with Trunk)
//! backed by a `yew_agent` `Worker` (`worker` submodule) that owns a
//! `worker::Model` enum over float (`m::model::Whisper`) or GGUF-quantized
//! (`m::quantized_model::Whisper`) checkpoints, plus `audio` (mel
//! spectrogram extraction) and `languages` (language-token table) helpers.
//! A second, JS-driven path (`src/bin/m.rs`) wraps the same decoder logic in
//! a `#[wasm_bindgen]` `Decoder::new(weights, tokenizer, mel_filters,
//! config, quantized, is_multilingual, timestamps, task, language)` +
//! `decode(wav_bytes) -> JSON segments`, for a plain WebWorker built via
//! `build-lib.sh`. Weights/tokenizer/mel filters are `wget`-ed from
//! `openai/whisper-tiny*` and `lmz/candle-whisper` (Trunk build) or fetched
//! by JS and passed in (WebWorker build).
pub const WITH_TIMER: bool = true;

mod app;
mod audio;
pub mod languages;
pub mod worker;
pub use app::App;
pub use worker::Worker;
