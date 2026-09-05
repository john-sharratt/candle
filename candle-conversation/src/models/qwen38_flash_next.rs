//! Qwen3.8-Flash-Next (sparse-attention MoE) model preset.
//!
//! 250 B total, ~13 B active, 48 layers at 3:1 — 36 gated-DeltaNet layers
//! carrying a recurrent state to 12 attention layers carrying paged K/V — plus
//! two carried classes no other model in this catalogue has: a **PLE** window
//! (conv history + hash window over a disk-resident embedding table) and a
//! **QSA index** (one pooled key per 4-token block, per attention layer, derived
//! from hidden states rather than from stored K).
//!
//! Those two are why the turn record carries a model-opaque blob beside the
//! delta-rule layers: the index cannot be rebuilt from restored K/V at any price
//! short of a full forward, so it is persisted. See
//! `docs/qwen38_index_persistence.md`.
//!
//! Native context is 262,144 tokens, and it is genuinely backed — measured flat
//! from 32K to 128K, 1,266 t/s bulk at both.
//!
//! # The engine GGUF is built here, not downloaded
//!
//! Unlike every other preset, no repo publishes this file. It is assembled from
//! two pinned releases:
//!
//! - `unsloth/Qwen3.8-Flash-Next-GGUF` — the plain **Q8_0** split, six shards.
//!   Q8_0 and not a smaller quant because both community sub-8-bit conversions
//!   carry IQ-family tensors this codebase cannot read (dtype codes 20/21,
//!   verified from the shard headers).
//! - `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` — the expert tensors, imported to
//!   Q4_KO bit-exactly (both are per-128 symmetric affine), which carries the
//!   release's AWQ calibration into the resident expert format.
//!
//! plus the MTP draft head folded in as `blk.{num_layers}`. The result is one
//! mmap and one `Content`, which is what the engine's loader takes.
//!
//! [`ModelSpec::prepared_from_source`] marks that, so resolution looks locally
//! and reports the prepare step instead of asking the hub for a filename nobody
//! published.

use super::{ModelArch, ModelSpec};
use crate::{config::SamplingConfig, models::DialectType};
use candle_transformers::models::quantized_qwen38_moe;

const PROMPT: &str = "You are a helpful, accurate, and concise assistant.";

/// The merged engine artifact `qwen4exp::convert` produces.
pub const ENGINE_GGUF: &str = "Qwen3.8-Flash-Next-Q4KOEXP-merged.gguf";

/// Qwen3.8-Flash-Next, Q4_KO experts over a Q8_0 trunk.
///
/// Resident footprint is the dense weights plus whatever expert working set
/// fits — ~54 GB on a 72 GB card, with the three-tier expert cache paging the
/// rest. As with every MoE here, parameter count does not decide feasibility.
pub(super) fn qwen38_flash_next_q4ko() -> ModelSpec {
    // Qwen3.5/3.8 share ChatML's markers but suppress thinking by opening the
    // assistant turn with an already-closed think block — Qwen3's `/no_think`
    // marker does nothing on this lineage.
    let chat_format = DialectType::Qwen35;
    ModelSpec {
        arch: ModelArch::Qwen4Exp,
        dialect: chat_format.dialect(),
        chat_format,
        // The repo the engine artifact is BUILT FROM. Nothing here publishes
        // `model_filename`; see the module docs.
        model_repo: quantized_qwen38_moe::QWEN4EXP_REPO.into(),
        model_filename: ENGINE_GGUF.into(),
        prepared_from_source: true,
        // A locally built artifact has no published length to pin, and the
        // merge's exact size depends on which expert format was requantized.
        // Zero means "no length check", which is correct here and would not be
        // for a published file.
        model_bytes: 0,
        // The gate's own pins, not copies. `quantized_qwen38_moe` is where the
        // claim "this tokenizer and the GGUF's `tokenizer.ggml.tokens` agree
        // token for token" was established; serving and gating reading the same
        // two constants is what keeps that claim true of what actually runs —
        // and this vocabulary is not Qwen3's (`<|im_end|>` is 248046 here).
        tokenizer_repo: quantized_qwen38_moe::TOKENIZER_REPO.into(),
        tokenizer_rev: quantized_qwen38_moe::TOKENIZER_REV.into(),
        default_system_prompt: PROMPT.into(),
        // `qwen4exp.context_length` in the merged engine GGUF, and backed:
        // the depth gate runs 32K and 128K at the same bulk rate.
        max_seq_len: 262_144,
        default_sampling: SamplingConfig::for_gguf_architecture("qwen2moe"),
        supports_thinking: true,
        non_thinking_sampling: SamplingConfig::non_thinking_for_gguf_architecture("qwen2moe"),
    }
}
