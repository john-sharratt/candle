//! Qwen3.6-35B-A3B (hybrid MoE) model preset.
//!
//! The hybrid lineage's entry into the conversation layer. Three quarters of
//! this stack's 40 layers mix tokens through a **recurrent state** rather than a
//! KV cache — 30 gated-DeltaNet layers to 10 attention layers, at 3:1 — which is
//! what the recurrent-state persistence work in
//! `docs/deltanet_state_persistence.md` exists to support.
//!
//! Geometry: 40 layers, 16 Q / 2 KV heads at `head_dim` 256, DeltaNet 32 V / 16
//! QK heads at 128, 256 experts top-8 plus a gated shared expert, hidden 2048.
//! The GGUF carries the `qwen35moe` arch string — Qwen3.6 is a point release of
//! the Qwen3.5 architecture and shares its metadata keys and tensor schema, so
//! both load through the same [`ModelArch::Qwen35Hybrid`] arm.

use super::{ModelArch, ModelSpec};
use crate::{config::SamplingConfig, models::DialectType};

const PROMPT: &str = "You are a helpful, accurate, and concise assistant.";

/// Qwen3.6-35B-A3B UD-Q4_K_M — 35 B total, ~3 B active.
///
/// Runs on a sub-24 GB card through the three-tier expert cache: the resident
/// footprint is the dense weights plus whatever expert working set fits, not the
/// parameter count.
pub(super) fn qwen36_35b_a3b_q4() -> ModelSpec {
    let chat_format = DialectType::ChatML;
    ModelSpec {
        arch: ModelArch::Qwen35Hybrid,
        dialect: chat_format.dialect(),
        chat_format,
        // The `-MTP-` repo, not the plain one. Both publish the same quant under
        // the same filename, and they differ only in that the plain conversion
        // drops the NextN tensors — so a model loaded from it cannot speculate.
        // That failure is silent: speculation is lossless, so a drafter-less
        // checkpoint answers identically and only the ~2x decode is missing.
        // `quantized_qwen36_moe` pins the same repo and asserts `has_drafter()`
        // for exactly this reason.
        model_repo: "unsloth/Qwen3.6-35B-A3B-MTP-GGUF".into(),
        model_filename: "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf".into(),
        // The published file's exact length, read off the local snapshot at the
        // pinned revision. Downloaders use it for progress totals when the
        // server omits Content-Length, so a guess shows a wrong bar. The MTP
        // file is the larger of the two by the size of the head.
        model_bytes: 22_663_387_424,
        tokenizer_repo: "Qwen/Qwen3.6-35B-A3B".into(),
        default_system_prompt: PROMPT.into(),
        max_seq_len: 4096,
        default_sampling: SamplingConfig::for_gguf_architecture("qwen2moe"),
        supports_thinking: true,
        non_thinking_sampling: SamplingConfig::non_thinking_for_gguf_architecture("qwen2moe"),
    }
}
