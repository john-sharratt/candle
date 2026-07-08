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
        // Original (April 2025) Qwen3-30B-A3B — the HYBRID model: its chat
        // template carries `<think>` + `enable_thinking`, so it honours the
        // `/think` ↔ `/no_think` soft switch the composer effort dial drives.
        // (The 2507 refresh split this into separate Instruct/Thinking models,
        // neither of which can toggle — see the spec history.)
        model_repo: "unsloth/Qwen3-30B-A3B-GGUF".into(),
        model_filename: "Qwen3-30B-A3B-Q4_K_M.gguf".into(),
        tokenizer_repo: "Qwen/Qwen3-30B-A3B".into(),
        default_system_prompt: PROMPT.into(),
        max_seq_len: 4096,
        default_sampling: SamplingConfig::for_gguf_architecture("qwen2moe"),
        supports_thinking: true,
        // Sampling params for a thinking-off turn (effort=Off / `/no_think`); the
        // empty `<think></think>` itself comes from the glue + Off steering.
        non_thinking_sampling: SamplingConfig::non_thinking_for_gguf_architecture("qwen2moe"),
    }
}
