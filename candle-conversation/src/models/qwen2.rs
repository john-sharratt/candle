//! Qwen2 model family — lightweight instruct-tuned variants.
//!
//! Uses [`ChatML`](super::ChatFormat::ChatML) format with `<|im_end|>` EOS
//! and official default sampling (temperature=0.7, top_p=0.8, top_k=20,
//! repetition_penalty=1.1).
//!
//! Source: <https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/raw/main/generation_config.json>

use super::{ModelArch, ModelSpec};
use crate::{config::SamplingConfig, models::DialectType};

// ────────────────────────────────────────────────────────────────────────────
// Qwen2-0.5B-Instruct
// ────────────────────────────────────────────────────────────────────────────

/// Qwen2-0.5B-Instruct Q4_0 — tiny, great for CI and testing (~0.4 GB VRAM).
pub(super) fn qwen2_0_5b() -> ModelSpec {
    let chat_format = DialectType::ChatML;
    ModelSpec {
        arch: ModelArch::Qwen2,
        dialect: chat_format.dialect(),
        chat_format,
        model_repo: "Qwen/Qwen2-0.5B-Instruct-GGUF".into(),
        model_filename: "qwen2-0_5b-instruct-q4_0.gguf".into(),
        tokenizer_repo: "Qwen/Qwen2-0.5B-Instruct".into(),
        default_system_prompt: "You are a helpful assistant.".into(),
        max_seq_len: 4096,
        default_sampling: SamplingConfig::for_gguf_architecture("qwen2"),
        supports_thinking: false,
        non_thinking_sampling: None,
    }
}
