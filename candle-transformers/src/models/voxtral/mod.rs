//! Voxtral speech-to-text model: mel audio encoder + Llama-based decoder.
//!
//! [`audio`] extracts log-mel features from raw audio (`N_FFT`/`HOP_LENGTH`/
//! `N_MELS` below fix the STFT parameters); [`model`] implements the
//! Whisper-style `VoxtralEncoder` and the `VoxtralForConditionalGeneration`
//! wrapper that projects encoder output into the decoder's embedding space
//! via `VoxtralMultiModalProjector`; [`voxtral_llama`] is the Llama decoder
//! variant used as the language backbone.
pub mod audio;
pub mod model;
pub mod voxtral_llama;

pub use audio::extract_features;
pub use model::{
    VoxtralCache, VoxtralConfig, VoxtralEncoder, VoxtralEncoderConfig,
    VoxtralForConditionalGeneration, VoxtralGenerationConfig, VoxtralMultiModalProjector,
};
pub use voxtral_llama::{VoxtralLlama, VoxtralLlamaCache, VoxtralLlamaConfig};

pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 128;
