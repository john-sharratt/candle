//! Qwen3-30B-A3B (MoE) model preset.
//!
//! Uses [`ChatML`](super::ChatFormat::ChatML) format with official Qwen3
//! sampling from [`SamplingConfig::for_gguf_architecture`].
//!
//! Source: <https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/generation_config.json>

use super::{ModelArch, ModelSpec};
use crate::{config::SamplingConfig, models::DialectType};

const PROMPT: &str = "You are a helpful, accurate, and concise assistant.";

// ────────────────────────────────────────────────────────────────────────────
// Qwen3-30B-A3B (MoE)
// ────────────────────────────────────────────────────────────────────────────

/// Qwen3-30B-A3B Q4_K_M — MoE with 128 experts, 8 active per token (~16 GB VRAM with LRU).
pub(super) fn qwen3_30b_a3b_q4() -> ModelSpec {
    let chat_format = DialectType::ChatML;
    ModelSpec {
        arch: ModelArch::Qwen3Moe,
        dialect: chat_format.dialect(),
        chat_format,
        model_repo: "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF".into(),
        model_filename: "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf".into(),
        tokenizer_repo: "Qwen/Qwen3-30B-A3B-Instruct-2507".into(),
        default_system_prompt: PROMPT.into(),
        max_seq_len: 4096,
        default_sampling: SamplingConfig::for_gguf_architecture("qwen2moe"),
        supports_thinking: true,
        non_thinking_sampling: SamplingConfig::non_thinking_for_gguf_architecture("qwen2moe"),
    }
}
