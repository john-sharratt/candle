//! Hermes-3 model family — ChatML-tuned Llama models from NousResearch.
//!
//! All variants use [`ChatML`](super::ChatFormat::ChatML) format (despite the
//! Llama base) with `<|im_end|>` EOS and sampling defaults from
//! [`SamplingConfig::for_gguf_architecture`].
//!
//! Source: <https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B/raw/main/generation_config.json>

use super::{ModelArch, ModelSpec};
use crate::{config::SamplingConfig, models::DialectType};

// ────────────────────────────────────────────────────────────────────────────
// Hermes-3-Llama-3.2-3B
// ────────────────────────────────────────────────────────────────────────────

/// Hermes-3-Llama-3.2-3B Q6_K — fast, ChatML-tuned (~3 GB VRAM).
pub(super) fn hermes3_3b_q6() -> ModelSpec {
    let chat_format = DialectType::Llama3;
    ModelSpec {
        arch: ModelArch::Llama,
        dialect: chat_format.dialect(),
        chat_format,
        model_repo: "bartowski/Hermes-3-Llama-3.2-3B-GGUF".into(),
        model_filename: "Hermes-3-Llama-3.2-3B-Q6_K.gguf".into(),
        model_bytes: 2_643_850_336,
        tokenizer_repo: "NousResearch/Hermes-3-Llama-3.2-3B".into(),
        default_system_prompt: "You are a helpful assistant.".into(),
        max_seq_len: 8192,
        default_sampling: SamplingConfig::for_gguf_architecture("llama"),
        supports_thinking: false,
        non_thinking_sampling: None,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Hermes-3-Llama-3.1-70B
// ────────────────────────────────────────────────────────────────────────────

/// Hermes-3-Llama-3.1-70B Q4_K_M — large, needs ≥48 GB VRAM (~40 GB).
pub(super) fn hermes3_70b_q4() -> ModelSpec {
    let chat_format = DialectType::Llama3;
    ModelSpec {
        arch: ModelArch::Llama,
        dialect: chat_format.dialect(),
        chat_format,
        model_repo: "bartowski/Hermes-3-Llama-3.1-70B-GGUF".into(),
        model_filename: "Hermes-3-Llama-3.1-70B-Q4_K_M.gguf".into(),
        model_bytes: 42_520_393_792,
        tokenizer_repo: "NousResearch/Hermes-3-Llama-3.1-70B".into(),
        default_system_prompt: "You are a helpful assistant.".into(),
        max_seq_len: 8192,
        default_sampling: SamplingConfig::for_gguf_architecture("llama"),
        supports_thinking: false,
        non_thinking_sampling: None,
    }
}
