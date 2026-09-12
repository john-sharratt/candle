//! Z-Image — Tongyi-MAI's text-to-image rectified-flow transformer.
//!
//! - 🤗 [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
//!
//! # The pipeline, and what candle already had
//!
//! Four models, of which only the transformer is new here:
//!
//! | Piece | What it is |
//! |---|---|
//! | Text encoder | **Qwen3-4B**, taken at `hidden_states[-2]` — [`text_encoder`], a GGUF-named, int8 reading of [`crate::models::qwen3`]'s architecture |
//! | Transformer | `ZImageTransformer2DModel` — [`model`], the only new architecture |
//! | Autoencoder | The FLUX VAE, verbatim. Its config even says `"_name_or_path": "flux-dev"`: 16 latent channels, scale 0.3611, shift 0.1159 — but published under *diffusers* tensor names, so it loads through [`crate::models::stable_diffusion::vae`] rather than [`crate::models::flux::autoencoder`] |
//! | Scheduler | Flow-match Euler with a fixed shift — [`sampling`] |
//!
//! # The transformer, in one paragraph
//!
//! It is NextDiT rather than a FLUX-style double/single-stream model. Image
//! patches and caption tokens are refined *separately* — two `noise_refiner`
//! blocks over the patches and two `context_refiner` blocks over the caption —
//! and then concatenated into one sequence that thirty full-width blocks attend
//! over jointly. Every block is a plain self-attention block: there is no cross
//! attention anywhere, because the caption is simply part of the sequence.
//!
//! Three details set it apart from the DiTs candle already carries:
//!
//! - **Sandwich norms.** Each sublayer is normed on the way in *and* on the way
//!   out (`attention_norm1`/`attention_norm2`), rather than pre-norm alone.
//! - **3-axis RoPE.** Position is a triple, not an index: `(t, h, w)` with 32,
//!   48 and 48 of the 128 head dimensions each. A caption token sits at
//!   `(1 + j, 0, 0)` and an image patch at `(cap_len + 1, h, w)`, so the text
//!   occupies the axis the image does not use.
//! - **Time enters as a scale, not a token.** `adaLN_modulation` produces four
//!   vectors per block from the timestep embedding — two scales applied before
//!   each sublayer, two `tanh` gates applied after. The context refiner has no
//!   modulation at all, because refining a caption does not depend on how far
//!   through the denoise we are.

//! # Both halves run int8
//!
//! The transformer and the text encoder are each loaded from a GGUF and each
//! projection repacked at load into its KO twin, so every matmul in an image —
//! prompt encode and all eight denoise steps — is q8a128 against a KO weight on
//! the tensor cores. See [`quant_choice`] for why that costs nothing at these
//! rungs, and [`quantized_model`] for the one projection too narrow to tile.

pub mod adapter;
pub mod attention;
#[cfg(feature = "cuda")]
pub mod attention_int8;
pub mod model;
pub mod quant_choice;
pub mod quantized_model;
pub mod sampling;
pub mod text_encoder;

pub use model::{Config, ZImageTransformer};
pub use quant_choice::{ZQuant, TEXT_ENCODER_FILE, TEXT_ENCODER_REPO};
pub use text_encoder::TextEncoder;
