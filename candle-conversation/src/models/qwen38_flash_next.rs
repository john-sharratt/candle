//! Qwen3.8-Flash-Next (sparse-attention MoE) model presets — one per KO expert
//! rung.
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
//! `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16`, or `Q3_KO` / `Q2_KO` requantized from
//! the `Q8_0` split (`candle_transformers::models::quant_ladder`). The result is
//! one mmap and one `Content`, which is what the engine's loader takes, named by
//! its recipe's digest (`qwen4exp::prepare`).
//!
//! Each rung is its own preset because each is its own artifact: the file a
//! preset names is the one its recipe builds, so a machine loads exactly what
//! its rung's recipe produced and nothing older.
//!
//! [`ModelSpec::prepared_from_source`] marks that, so resolution looks locally
//! and reports the prepare step instead of asking the hub for a filename nobody
//! published.

use super::{ModelArch, ModelSpec, RopePreset};
use crate::{config::SamplingConfig, models::DialectType};
use candle::quantized::GgmlDType;
use candle_transformers::models::quantized_qwen38_moe;

const PROMPT: &str = "You are a helpful, accurate, and concise assistant.";

/// The engine artifact of the rung whose routed experts are `experts`, named by
/// its recipe (`quantized_qwen38_moe::engine_recipe_at`), so a change to what
/// the build would produce names a different file and the old one is not loaded.
pub fn engine_gguf(experts: GgmlDType) -> String {
    quantized_qwen38_moe::engine_recipe_at(Some(experts)).artifact_name()
}

/// Qwen3.8-Flash-Next, `Q4_KO` experts over a `Q8_0` trunk — the rung for cards
/// of 64 GiB and up.
///
/// Resident footprint is the dense weights plus whatever expert working set
/// fits — ~54 GB on a 72 GB card, with the three-tier expert cache paging the
/// rest. As with every MoE here, parameter count does not decide feasibility.
pub(super) fn qwen38_flash_next_q4ko() -> ModelSpec {
    flash_next(GgmlDType::Q4_KO)
}

/// Qwen3.8-Flash-Next, `Q3_KO` experts — the rung for 32–63 GiB cards.
pub(super) fn qwen38_flash_next_q3ko() -> ModelSpec {
    flash_next(GgmlDType::Q3_KO)
}

/// Qwen3.8-Flash-Next, `Q2_KO` experts — the rung for cards under 32 GiB, the
/// 16 GB laptop among them.
pub(super) fn qwen38_flash_next_q2ko() -> ModelSpec {
    flash_next(GgmlDType::Q2_KO)
}

/// The preset whose artifact carries `experts`.
fn flash_next(experts: GgmlDType) -> ModelSpec {
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
        model_filename: engine_gguf(experts),
        prepared_from_source: true,
        // Nothing to pin: the engine GGUF is built locally from
        // `model_repo`'s published files, so there is no upstream commit that
        // names these bytes. The recipe pins the build's inputs, and its digest
        // is in the filename above.
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
    use crate::models::builder::model_cache_dir;

    /// **Each rung names the artifact its own recipe builds**, and the three
    /// are distinct files — a card never loads another rung's experts.
    #[test]
    fn each_rung_names_its_recipes_artifact() {
        let q2 = qwen38_flash_next_q2ko().model_filename;
        let q3 = qwen38_flash_next_q3ko().model_filename;
        let q4 = qwen38_flash_next_q4ko().model_filename;
        assert_eq!(q2, "Qwen3.8-Flash-Next-Q2_KOEXP-130076148f33.gguf");
        assert!(q3.starts_with("Qwen3.8-Flash-Next-Q3_KOEXP-"), "{q3}");
        assert!(q4.starts_with("Qwen3.8-Flash-Next-Q4_KOEXP-"), "{q4}");
        assert!(q2 != q3 && q3 != q4 && q2 != q4);
    }

    /// **The gate builds where the loader looks.** `quantized_qwen38_moe` sits
    /// below this crate and cannot call [`model_cache_dir`], so it repeats the
    /// rule; this pins the two to the same directory. They once disagreed — the
    /// gate honoured `XDG_CACHE_HOME` and the loader did not — so an artifact
    /// built by the gate could be reported missing by the probe beside it.
    #[test]
    fn the_gates_artifact_dir_is_the_loaders() {
        let spec = qwen38_flash_next_q2ko();
        assert_eq!(
            quantized_qwen38_moe::engine_artifact_dir(),
            model_cache_dir().join(spec.model_repo.replace('/', "--"))
        );
    }
}
