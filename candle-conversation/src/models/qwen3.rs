//! Qwen3 model family — 8B and 14B variants across multiple quantisations.
//!
//! All variants use [`ChatML`](super::ChatFormat::ChatML) format,
//! `<|im_end|>` as the EOS token, and sampling defaults from
//! [`SamplingConfig::for_gguf_architecture`].

use super::{ModelArch, ModelSpec};
use crate::{config::SamplingConfig, models::DialectType};

const PROMPT: &str = "You are a helpful, accurate, and concise assistant.";

// ────────────────────────────────────────────────────────────────────────────
// Qwen3-8B
// ────────────────────────────────────────────────────────────────────────────

/// Qwen3-8B Q4_K_M — strong general-purpose 8B model (~5 GB VRAM).
pub(super) fn qwen3_8b_q4() -> ModelSpec {
    let chat_format = DialectType::ChatML;
    ModelSpec {
        arch: ModelArch::Qwen3,
        dialect: chat_format.dialect(),
        chat_format,
        model_repo: "unsloth/Qwen3-8B-GGUF".into(),
        model_filename: "Qwen3-8B-Q4_K_M.gguf".into(),
        tokenizer_repo: "Qwen/Qwen3-8B".into(),
        default_system_prompt: PROMPT.into(),
        max_seq_len: 8192,
        default_sampling: SamplingConfig::for_gguf_architecture("qwen3"),
        supports_thinking: true,
        non_thinking_sampling: SamplingConfig::non_thinking_for_gguf_architecture("qwen3"),
    }
}

/// Qwen3-8B Q6_K — higher-quality 8B quantisation (~7 GB VRAM).
pub(super) fn qwen3_8b_q6() -> ModelSpec {
    let chat_format = DialectType::ChatML;
    ModelSpec {
        arch: ModelArch::Qwen3,
        dialect: chat_format.dialect(),
        chat_format,
        model_repo: "unsloth/Qwen3-8B-GGUF".into(),
        model_filename: "Qwen3-8B-Q6_K.gguf".into(),
        tokenizer_repo: "Qwen/Qwen3-8B".into(),
        default_system_prompt: PROMPT.into(),
        max_seq_len: 8192,
        default_sampling: SamplingConfig::for_gguf_architecture("qwen3"),
        supports_thinking: true,
        non_thinking_sampling: SamplingConfig::non_thinking_for_gguf_architecture("qwen3"),
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Qwen3-14B
// ────────────────────────────────────────────────────────────────────────────

/// Qwen3-14B Q4_K_M — strong 14B model (~8 GB VRAM).
pub(super) fn qwen3_14b_q4() -> ModelSpec {
    let chat_format = DialectType::ChatML;
    ModelSpec {
        arch: ModelArch::Qwen3,
        dialect: chat_format.dialect(),
        chat_format,
        model_repo: "unsloth/Qwen3-14B-GGUF".into(),
        model_filename: "Qwen3-14B-Q4_K_M.gguf".into(),
        tokenizer_repo: "Qwen/Qwen3-14B".into(),
        default_system_prompt: PROMPT.into(),
        max_seq_len: 8192,
        default_sampling: SamplingConfig::for_gguf_architecture("qwen3"),
        supports_thinking: true,
        non_thinking_sampling: SamplingConfig::non_thinking_for_gguf_architecture("qwen3"),
    }
}

/// Qwen3-14B Q5_K_M — balanced quality/size (~10 GB VRAM).
pub(super) fn qwen3_14b_q5() -> ModelSpec {
    let chat_format = DialectType::ChatML;
    ModelSpec {
        arch: ModelArch::Qwen3,
        dialect: chat_format.dialect(),
        chat_format,
        model_repo: "unsloth/Qwen3-14B-GGUF".into(),
        model_filename: "Qwen3-14B-Q5_K_M.gguf".into(),
        tokenizer_repo: "Qwen/Qwen3-14B".into(),
        default_system_prompt: PROMPT.into(),
        max_seq_len: 8192,
        default_sampling: SamplingConfig::for_gguf_architecture("qwen3"),
        supports_thinking: true,
        non_thinking_sampling: SamplingConfig::non_thinking_for_gguf_architecture("qwen3"),
    }
}

/// Qwen3-14B Q6_K — near-FP16 quality (~12 GB VRAM).
pub(super) fn qwen3_14b_q6() -> ModelSpec {
    let chat_format = DialectType::ChatML;
    ModelSpec {
        arch: ModelArch::Qwen3,
        dialect: chat_format.dialect(),
        chat_format,
        model_repo: "unsloth/Qwen3-14B-GGUF".into(),
        model_filename: "Qwen3-14B-Q6_K.gguf".into(),
        tokenizer_repo: "Qwen/Qwen3-14B".into(),
        default_system_prompt: PROMPT.into(),
        max_seq_len: 8192,
        default_sampling: SamplingConfig::for_gguf_architecture("qwen3"),
        supports_thinking: true,
        non_thinking_sampling: SamplingConfig::non_thinking_for_gguf_architecture("qwen3"),
    }
}
