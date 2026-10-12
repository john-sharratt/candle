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
//! Unlike every other preset, no repo publishes this file. It is assembled on
//! each machine from pinned releases, as a hybrid: the `Q8_0` trunk and n-gram
//! table of `unsloth/Qwen3.8-Flash-Next-GGUF` verbatim, the MTP draft head
//! (`MTP/` of the same repo) folded in as `blk.{num_layers}`, and the routed
//! experts at the width of the card's rung — `Q4_KO` imported bit-exactly from
//! `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` on a card of 64 GiB and up, `Q3_KO`
//! requantized from the `Q8_0` split below that
//! (`candle_transformers::models::quant_ladder`). The artifact is named by its
//! recipe's digest (`qwen4exp::prepare`), and the engine loads the model pack
//! built from it — which releases the artifact once it holds it.
//!
//! # One model on every card
//!
//! The rung is the card's, not the preset's: the one preset loads whichever
//! rung's pack this card's VRAM calls for. Every rung holds the same release,
//! so the model is named by the release's trunk and draft-head sources
//! (`quantized_qwen38_moe::engine_model_digest`) rather than by the artifact a
//! card built, and a substrate written on one card opens on another.
//!
//! [`ModelSpec::prepared_from_source`] marks that the artifact is prepared, so
//! resolution builds the pack only from an artifact already prepared on this
//! machine (`quantized_qwen38_moe::prepared_engine_pack`) instead of asking the
//! hub for a filename nobody published.

use super::{ModelArch, ModelSpec, RopePreset};
use crate::{config::SamplingConfig, models::DialectType};
use candle_transformers::models::quantized_qwen38_moe;

const PROMPT: &str = "You are a helpful, accurate, and concise assistant.";

/// The name the model goes by: the release's, with no rung in it, since the
/// rung a card loads is that card's.
pub const MODEL_NAME: &str = "Qwen3.8-Flash-Next.gguf";

/// Qwen3.8-Flash-Next at this card's rung.
///
/// Resident footprint is the dense weights plus whatever expert working set
/// fits — ~54 GB on a 72 GB card, with the three-tier expert cache paging the
/// rest. As with every MoE here, parameter count does not decide feasibility.
pub(super) fn qwen38_flash_next() -> ModelSpec {
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
        model_filename: MODEL_NAME.into(),
        prepared_from_source: true,
        // Nothing to pin: the engine GGUF is built locally from
        // `model_repo`'s published files, so there is no upstream commit that
        // names these bytes. The recipe pins the build's inputs, and its digest
        // is in the artifact's filename.
        model_rev: String::new(),
        // Used as prepared — no override displaced a preset here, so there is no
        // base checkpoint to fall back to for tensors this one got wrong.
        gate_donor: None,
        tensor_overrides: Vec::new(),
        // No adapter ships with this model. An adapter is opt-in per
        // conversation; an empty list is what makes the base model the default.
        loras: Vec::new(),
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
        // `qwen4exp.context_length` in the engine artifact, and backed: the
        // depth gate runs 32K and 128K at the same bulk rate.
        max_seq_len: 262_144,
        rope: RopePreset::Lineage,
        default_sampling: SamplingConfig::for_gguf_architecture("qwen2moe"),
        supports_thinking: true,
        non_thinking_sampling: SamplingConfig::non_thinking_for_gguf_architecture("qwen2moe"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_transformers::models::model_pack::cache_root;

    /// **The preset names the release, not a rung**: no expert width in its
    /// name, so the same preset is the model on every card.
    #[test]
    fn the_preset_names_the_release() {
        let spec = qwen38_flash_next();
        assert_eq!(spec.model_filename, "Qwen3.8-Flash-Next.gguf");
        assert!(spec.prepared_from_source);
    }

    /// **The gate prepares where the loader looks**: the artifact lands in the
    /// model cache's directory for its source repo, the one the engine pack is
    /// filed under. They once disagreed — the gate honoured `XDG_CACHE_HOME` and
    /// the loader did not — so an artifact built by the gate could be reported
    /// missing by the probe beside it.
    #[test]
    fn the_gates_artifact_dir_is_the_loaders() {
        let spec = qwen38_flash_next();
        assert_eq!(
            quantized_qwen38_moe::engine_artifact_dir(),
            cache_root().join(spec.model_repo.replace('/', "--"))
        );
    }
}
