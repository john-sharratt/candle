//! Pre-configured model presets with a builder pattern for streamlined setup.
//!
//! The [`Model`] enum provides strongly-typed presets (HF coordinates, chat
//! format, EOS token, default sampling). Call [`.builder()`](Model::builder)
//! to customise any setting before constructing a
//! [`ConversationEngine`](crate::ConversationEngine).
//!
//! # Quick Start
//!
//! ```ignore
//! use candle_conversation::models::Model;
//!
//! let device = candle::Device::cuda_if_available(0)?;
//!
//! // One-liner (downloads from HuggingFace, requires `hub` feature):
//! let engine = Model::Qwen3_8B_Q4.engine(&device)?;
//!
//! // With customisation:
//! let b = Model::Qwen3_14B_Q4.builder()
//!     .temperature(0.8)
//!     .max_response_tokens(4096);
//! let engine = b.engine(&device)?;
//! let mut conv = engine.new_conversation(&b.system_prompt(), b.conversation_config())?;
//! let resp = conv.send("Hello!")?;
//! ```
//!
//! # Local Files
//!
//! ```ignore
//! // Directory containing the GGUF and tokenizer.json:
//! let engine = Model::Qwen3_8B_Q4.builder()
//!     .model_dir("/models/qwen3-8b")
//!     .engine(&device)?;
//!
//! // Or explicit paths:
//! let engine = Model::Qwen3_8B_Q4.builder()
//!     .model_path("/models/Qwen3-8B-Q4_K_M.gguf")
//!     .tokenizer_path("/models/tokenizer.json")
//!     .engine(&device)?;
//! ```
//!
//! # MoE Support
//!
//! **Qwen3-30B-A3B (MoE)** is supported via `quantized_qwen3_moe` with an LRU
//! expert cache and `cudaHostAllocMapped` for non-expert weight overflow.
//! Requires a CUDA GPU with ≥16 GB VRAM.

mod builder;
mod dialect;
mod hermes3;
mod qwen2;
mod qwen3;
mod qwen3_moe;

pub use builder::ModelBuilder;
pub use dialect::*;

use crate::config::{SequenceConfig, SamplingConfig};
use crate::error::ConversationError;
use std::path::Path;

// ────────────────────────────────────────────────────────────────────────────
// Enums
// ────────────────────────────────────────────────────────────────────────────

/// Model architecture — selects the quantised weight loader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelArch {
    /// `quantized_qwen3::ModelWeights`
    Qwen3,
    /// `quantized_qwen3_moe::ModelWeights`
    Qwen3Moe,
    /// `quantized_qwen2::ModelWeights`
    Qwen2,
    /// `quantized_llama::ModelWeights`
    Llama,
}

/// Pre-configured model presets.
///
/// Each variant carries HF coordinates, architecture, chat format, and
/// recommended defaults. Call [`.builder()`](Model::builder) to customise,
/// or use convenience methods directly.
///
/// Use [`Model::custom`] (or [`ModelBuilder::from_spec`]) to bring your own
/// GGUF model that isn't in the preset list.
///
/// | Variant | Params | Quant | Arch | Format | VRAM |
/// |---|---|---|---|---|---|
/// | `Qwen3_8B_Q4` | 8 B | Q4_K_M | Qwen3 | ChatML | ~5 GB |
/// | `Qwen3_8B_Q6` | 8 B | Q6_K | Qwen3 | ChatML | ~7 GB |
/// | `Qwen3_14B_Q4` | 14 B | Q4_K_M | Qwen3 | ChatML | ~8 GB |
/// | `Qwen3_14B_Q5` | 14 B | Q5_K_M | Qwen3 | ChatML | ~10 GB |
/// | `Qwen3_14B_Q6` | 14 B | Q6_K | Qwen3 | ChatML | ~12 GB |
/// | `Qwen2_0_5B` | 0.5 B | Q4_0 | Qwen2 | ChatML | ~0.4 GB |
/// | `Hermes3_3B_Q6` | 3 B | Q6_K | Llama | ChatML | ~3 GB |
/// | `Hermes3_70B_Q4` | 70 B | Q4_K_M | Llama | ChatML | ~40 GB |
/// | `Qwen3_30B_A3B_Q4` | 30 B (3B active) | Q4_K_M | Qwen3Moe | ChatML | ~16 GB (LRU) |
/// | `Custom(_)` | — | — | any | any | — |
#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub enum Model {
    // ── Qwen3 ──────────────────────────────────────────────────────────
    /// Qwen3-8B Q4_K_M — strong general-purpose 8B model (~5 GB).
    Qwen3_8B_Q4,
    /// Qwen3-8B Q6_K — higher-quality 8B quantisation (~7 GB).
    Qwen3_8B_Q6,
    /// Qwen3-14B Q4_K_M — strong 14B model (~8 GB).
    Qwen3_14B_Q4,
    /// Qwen3-14B Q5_K_M — balanced quality/size (~10 GB).
    Qwen3_14B_Q5,
    /// Qwen3-14B Q6_K — near-FP16 quality (~12 GB).
    Qwen3_14B_Q6,

    // ── Qwen3 MoE ──────────────────────────────────────────────────────
    /// Qwen3-30B-A3B Q4_K_M — MoE, 128 experts, 8 active (~16 GB with LRU).
    Qwen3_30B_A3B_Q4,

    // ── Qwen2 ──────────────────────────────────────────────────────────
    /// Qwen2-0.5B-Instruct Q4_0 — tiny, great for CI and testing (~0.4 GB).
    Qwen2_0_5B,

    // ── Hermes-3 / Llama ───────────────────────────────────────────────
    /// Hermes-3-Llama-3.2-3B Q6_K — fast, ChatML-tuned (~3 GB).
    Hermes3_3B_Q6,
    /// Hermes-3-Llama-3.1-70B Q4_K_M — large, needs ≥48 GB VRAM (~40 GB).
    Hermes3_70B_Q4,

    // ── Custom ─────────────────────────────────────────────────────────
    /// User-provided model specification.
    ///
    /// Construct via [`Model::custom`]:
    ///
    /// ```ignore
    /// let spec = ModelSpec {
    ///     arch: ModelArch::Llama,
    ///     chat_format: DialectType::ChatML,
    ///     model_repo: "my-org/my-model-GGUF".into(),
    ///     model_filename: "my-model-Q4_K_M.gguf".into(),
    ///     tokenizer_repo: "my-org/my-model".into(),
    ///     eos_token: "<|im_end|>".into(),
    ///     default_system_prompt: "You are a helpful assistant.".into(),
    ///     max_seq_len: 8192,
    ///     default_sampling: SamplingConfig::top_p(0.9, 0.7),
    /// };
    /// let engine = Model::custom(spec)
    ///     .model_dir("/models/my-model")
    ///     .engine(&device)?;
    /// ```
    Custom(ModelSpec),
}

// ────────────────────────────────────────────────────────────────────────────
// ModelSpec
// ────────────────────────────────────────────────────────────────────────────

/// Immutable metadata for a model variant.
///
/// Returned by [`Model::spec`]. For built-in presets the string fields are
/// populated from static literals; for [`Model::Custom`] they can be any
/// owned [`String`].
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// Weight-loader architecture.
    pub arch: ModelArch,
    /// Chat template format.
    pub chat_format: DialectType,
    /// Dialect used to construct chat messages
    pub dialect: Dialect,
    /// HuggingFace repository containing the GGUF file.
    pub model_repo: String,
    /// GGUF filename within the repository.
    pub model_filename: String,
    /// HuggingFace repository containing `tokenizer.json`.
    pub tokenizer_repo: String,
    /// Default system prompt text (before chat-format wrapping).
    pub default_system_prompt: String,
    /// Maximum sequence length for KV cache allocation.
    pub max_seq_len: usize,
    /// Recommended default sampling strategy for this model family.
    pub default_sampling: SamplingConfig,
    /// Whether this model supports thinking/reasoning mode (`<think>` blocks).
    ///
    /// When `true` and the user has NOT enabled `--thinking`, the engine
    /// prefills an empty `<think></think>` block after the assistant header
    /// to suppress internal reasoning output.
    pub supports_thinking: bool,
    /// Alternate sampling parameters used when thinking is suppressed.
    ///
    /// Qwen3 recommends different settings for non-thinking mode:
    /// temperature=0.7, top_p=0.8, top_k=20.
    pub non_thinking_sampling: Option<SamplingConfig>,
}

// ────────────────────────────────────────────────────────────────────────────
// Model impl
// ────────────────────────────────────────────────────────────────────────────
impl Model {
    /// Create a [`Model::Custom`] from a user-provided [`ModelSpec`].
    ///
    /// Returns a [`ModelBuilder`] directly — no need to call `.builder()`.
    ///
    /// ```ignore
    /// let engine = Model::custom(spec)
    ///     .model_dir("/models/my-model")
    ///     .engine(&device)?;
    /// ```
    pub fn custom(spec: ModelSpec) -> ModelBuilder {
        ModelBuilder::from_spec(spec)
    }

    /// Full specification for this model variant.
    pub fn spec(self) -> ModelSpec {
        match self {
            // Qwen3
            Model::Qwen3_8B_Q4 => qwen3::qwen3_8b_q4(),
            Model::Qwen3_8B_Q6 => qwen3::qwen3_8b_q6(),
            Model::Qwen3_14B_Q4 => qwen3::qwen3_14b_q4(),
            Model::Qwen3_14B_Q5 => qwen3::qwen3_14b_q5(),
            Model::Qwen3_14B_Q6 => qwen3::qwen3_14b_q6(),
            // Qwen3 MoE
            Model::Qwen3_30B_A3B_Q4 => qwen3_moe::qwen3_30b_a3b_q4(),
            // Qwen2
            Model::Qwen2_0_5B => qwen2::qwen2_0_5b(),
            // Hermes-3 / Llama
            Model::Hermes3_3B_Q6 => hermes3::hermes3_3b_q6(),
            Model::Hermes3_70B_Q4 => hermes3::hermes3_70b_q4(),
            // Custom
            Model::Custom(spec) => spec,
        }
    }

    /// Create a [`ModelBuilder`] pre-loaded with this model's defaults.
    ///
    /// The builder lets you override sampling, file paths, sequence length,
    /// and other settings before calling [`.engine()`](ModelBuilder::engine).
    pub fn builder(self) -> ModelBuilder {
        ModelBuilder::from_spec(self.spec())
    }

    // ── Convenience shortcuts (delegate to a fresh builder) ────────────

    /// Shortcut: `self.builder().engine(device)`.
    ///
    /// Downloads from HuggingFace (requires `hub` feature) or returns an
    /// error if local paths are not configured.
    pub fn engine(self, device: &candle::Device) -> crate::Result<crate::ConversationEngine> {
        self.builder().engine(device)
    }

    /// Shortcut: `self.builder().conversation_config()`.
    pub fn conversation_config(self) -> SequenceConfig {
        self.builder().conversation_config()
    }

    /// Shortcut: `self.builder().system_prompt()`.
    pub fn default_system_prompt(self) -> String {
        self.builder().format_system_prompt()
    }

    /// Load a tokenizer from a local `tokenizer.json` file.
    pub fn load_tokenizer(path: &Path) -> crate::Result<tokenizers::Tokenizer> {
        tokenizers::Tokenizer::from_file(path)
            .map_err(|e| ConversationError::Tokenizer(e.to_string()))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Display
// ────────────────────────────────────────────────────────────────────────────

impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Model::Custom(spec) => {
                let name = spec
                    .model_filename
                    .strip_suffix(".gguf")
                    .unwrap_or(&spec.model_filename);
                write!(f, "Custom({name})")
            }
            other => {
                let spec = other.clone().spec();
                let name = spec
                    .model_filename
                    .strip_suffix(".gguf")
                    .unwrap_or(&spec.model_filename);
                write!(f, "{name}")
            }
        }
    }
}

impl std::fmt::Display for ModelArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelArch::Qwen3 => write!(f, "Qwen3"),
            ModelArch::Qwen3Moe => write!(f, "Qwen3Moe"),
            ModelArch::Qwen2 => write!(f, "Qwen2"),
            ModelArch::Llama => write!(f, "Llama"),
        }
    }
}
