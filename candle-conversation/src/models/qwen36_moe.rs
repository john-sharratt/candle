//! Qwen3.6-35B-A3B (hybrid MoE) model presets.
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

use super::{ModelArch, ModelSpec, RopePreset, TensorOverrideSpec};
use crate::config::{ModeSampling, SamplingConfig};
use crate::models::DialectType;
use candle_transformers::models::quantized_qwen36_moe;

const PROMPT: &str = "You are a helpful, accurate, and concise assistant.";

/// Qwen3.6-35B-A3B UD-Q4_K_M — 35 B total, ~3 B active.
///
/// Runs on a sub-24 GB card through the three-tier expert cache: the resident
/// footprint is the dense weights plus whatever expert working set fits, not the
/// parameter count.
///
/// # The dialect is the family's own
///
/// `Qwen35`, not `ChatML` — the finding [`super::qwen35_dense`] records for the
/// 9B holds for this checkpoint too. The two dialects agree on every turn
/// marker and differ in how reasoning is suppressed: ChatML prepends a
/// `/no_think` soft switch this family's template does not contain, so the
/// marker arrives as text and the turn reasons anyway, while the family honours
/// an already-closed think block prefilled after the assistant header. Under
/// ChatML every suppressed turn reasons into its budget — the titler's among
/// them, whose title then comes back empty.
pub(super) fn qwen36_35b_a3b_q4() -> ModelSpec {
    let chat_format = DialectType::Qwen35;
    ModelSpec {
        arch: ModelArch::Qwen35Hybrid,
        loras: Vec::new(),
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
        prepared_from_source: false,
        // The published file's exact length, read off the local snapshot at the
        // pinned revision. Downloaders use it for progress totals when the
        // server omits Content-Length, so a guess shows a wrong bar. The MTP
        // file is the larger of the two by the size of the head.
        model_bytes: 22_663_387_424,
        // The gate's own pin, not a copy of it. `quantized_qwen36_moe` is where
        // the claim "this tokenizer and the GGUF's `tokenizer.ggml.tokens`
        // agree token for token" was established; serving and gating reading
        // the same two constants is what stops them drifting apart, which is
        // the only way that claim stays true of what actually runs.
        // Unpinned: no verified commit has been recorded for this conversion. See
        // `ModelSpec::model_rev` — an empty revision resolves `main`, which moves.
        model_rev: String::new(),
        gate_donor: None,
        tensor_overrides: Vec::new(),
        tokenizer_repo: quantized_qwen36_moe::TOKENIZER_REPO.into(),
        tokenizer_rev: quantized_qwen36_moe::TOKENIZER_REV.into(),
        default_system_prompt: PROMPT.into(),
        max_seq_len: 4096,
        rope: RopePreset::Lineage,
        default_sampling: SamplingConfig::for_gguf_architecture("qwen2moe"),
        supports_thinking: true,
        non_thinking_sampling: SamplingConfig::non_thinking_for_gguf_architecture("qwen2moe"),
    }
}

/// The GGUF `general.architecture` string this lineage's routed checkpoints
/// carry — the key into [`SamplingConfig::for_gguf_architecture`], so it is the
/// sampling decision, not a label.
const ARCH: &str = "qwen35moe";

/// The `(temperature, top_p)` the hybrid decodes at, reasoning or not.
///
/// One pairing for both think modes, where the stock card publishes two (`1.0 / 0.95` while it
/// reasons, `0.7 / 0.8` while it does not): the cooler temperature with the wider nucleus.
/// `top_k 20` and `presence_penalty 1.5` are the lineage's own and arrive with the arch row.
///
/// A cast decodes on this only when a mission has it reason.
/// `SamplingConfig::for_character_dialogue` widens the think-off row for characters on top of
/// it — see `npcd::engine::mind`.
const HYBRID_SAMPLING: (f32, f32) = (0.7, 0.95);

/// **The hybrid — AntiLoop's trunk under StyleTune's output head** — `npcd`'s model.
///
/// Two fine-tunes of the checkpoint [`qwen36_35b_a3b_q4`] serves, both as mradermacher's static
/// Q4_K_M. Every tensor is AntiLoop's — a tune trained against the repetition loops a cast
/// falls into — except `output.weight`, which is StyleTune's, the tune for prose. The coordinates
/// are the gate's own (`quantized_qwen36_moe::QWEN36_35B_A3B_ANTILOOP` and
/// `QWEN36_35B_A3B_STYLETUNE`), so the daemon loads the bytes
/// `test_parallel_batched_forwarding_36_35b_antiloop_styletune` measured.
///
/// Everything that is a property of the architecture rather than of a file is shared with the
/// stock preset: the loader arm, the tokenizer the gate verified token for token against this
/// lineage's GGUFs, and the KV threshold row the loader applies (`QWEN36_MOE_KV_FACTORS`). One
/// row serves both files, so it is fitted to whichever sits nearer the edge: this hybrid on the
/// `Int8Mode::Performance` path, which needed a tighter K at C10×64 than the stock file does.
///
/// # Three files, and what each costs a fresh machine
///
/// AntiLoop is the checkpoint. StyleTune is fetched whole for its head: a tensor is read from a
/// mapped GGUF, and the hub serves files, not tensors. The stock checkpoint is the gate donor,
/// and it is needed: both mradermacher conversions store `ssm_alpha`/`ssm_beta` at Q4_K where
/// the stock file keeps them F32, and the loader refuses a quantized recurrent path it has no
/// base to repair from.
///
/// # The dialect is the family's own
///
/// `Qwen35`, not `ChatML`: the two agree on every turn marker, and ChatML's
/// `/no_think` soft switch is one this family's template does not contain, so
/// it would reach every conversation as a line of text. See
/// [`super::qwen35_dense`] for how that was found.
///
/// # Speculation
///
/// AntiLoop's conversion keeps the NextN head (`nextn_predict_layers = 1`), so the hybrid
/// drafts. StyleTune's does not, which is one reason the trunk is AntiLoop's.
pub(super) fn qwen36_35b_a3b_antiloop_styletune() -> ModelSpec {
    let chat_format = DialectType::Qwen35;
    let (repo, rev, file) = quantized_qwen36_moe::QWEN36_35B_A3B_ANTILOOP;
    let (donor_repo, donor_rev, donor_file) = quantized_qwen36_moe::QWEN36_35B_A3B;
    let (head_repo, head_rev, head_file) = quantized_qwen36_moe::QWEN36_35B_A3B_STYLETUNE;
    let modes = ModeSampling {
        thinking: HYBRID_SAMPLING,
        instruct: HYBRID_SAMPLING,
    };
    ModelSpec {
        arch: ModelArch::Qwen35Hybrid,
        loras: Vec::new(),
        dialect: chat_format.dialect(),
        chat_format,
        model_repo: repo.into(),
        model_filename: file.into(),
        prepared_from_source: false,
        model_bytes: 21_713_463_520,
        model_rev: rev.into(),
        gate_donor: Some((donor_repo.into(), donor_rev.into(), donor_file.into())),
        tensor_overrides: vec![TensorOverrideSpec {
            tensor: quantized_qwen36_moe::HYBRID_HEAD_TENSOR.into(),
            repo: head_repo.into(),
            revision: head_rev.into(),
            filename: head_file.into(),
        }],
        tokenizer_repo: quantized_qwen36_moe::TOKENIZER_REPO.into(),
        tokenizer_rev: quantized_qwen36_moe::TOKENIZER_REV.into(),
        default_system_prompt: PROMPT.into(),
        // Room for a character's history: the assembled prompt is large before
        // any of it, as it is for the dense 9B.
        max_seq_len: 8192,
        rope: RopePreset::Lineage,
        default_sampling: SamplingConfig::for_gguf_architecture(ARCH).with_mode_sampling(modes),
        supports_thinking: true,
        // The think-off config keeps its mode, so taking the pair adopts the instruct half.
        non_thinking_sampling: SamplingConfig::non_thinking_for_gguf_architecture(ARCH)
            .map(|s| s.with_mode_sampling(modes)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stencil::ThinkMode;

    /// **The gate's three checkpoints, pinned, and nothing else.** Serving and gating read the
    /// same constants, so the daemon loads the bytes the gate measured.
    #[test]
    fn the_hybrid_is_the_gates_three_pinned_checkpoints() {
        let s = qwen36_35b_a3b_antiloop_styletune();
        let (repo, rev, file) = quantized_qwen36_moe::QWEN36_35B_A3B_ANTILOOP;
        assert!(matches!(s.arch, ModelArch::Qwen35Hybrid), "{:?}", s.arch);
        assert_eq!(
            (
                s.model_repo.as_str(),
                s.model_rev.as_str(),
                s.model_filename.as_str()
            ),
            (repo, rev, file)
        );

        let (repo, rev, file) = quantized_qwen36_moe::QWEN36_35B_A3B_STYLETUNE;
        assert_eq!(
            s.tensor_overrides,
            vec![TensorOverrideSpec {
                tensor: "output.weight".into(),
                repo: repo.into(),
                revision: rev.into(),
                filename: file.into(),
            }],
            "the head, and only the head, is StyleTune's"
        );

        for pin in [&s.model_rev, &s.tensor_overrides[0].revision] {
            assert!(
                pin.len() == 40 && pin.chars().all(|c| c.is_ascii_hexdigit()),
                "{pin:?} is not a commit"
            );
        }
        assert_eq!(s.tokenizer_repo, quantized_qwen36_moe::TOKENIZER_REPO);
        assert_eq!(s.tokenizer_rev, quantized_qwen36_moe::TOKENIZER_REV);
        assert!(s.loras.is_empty());
        assert!(s.max_seq_len >= 8192);
    }

    /// **Locked on Q4_K_M** — the trunk and the file the head comes from alike.
    #[test]
    fn both_halves_are_q4_k_m() {
        let s = qwen36_35b_a3b_antiloop_styletune();
        assert!(
            s.model_filename.ends_with(".Q4_K_M.gguf"),
            "{}",
            s.model_filename
        );
        assert!(
            s.tensor_overrides[0].filename.ends_with(".Q4_K_M.gguf"),
            "{}",
            s.tensor_overrides[0].filename
        );
    }

    /// **The family's dialect, so no soft switch reaches a conversation as
    /// text.**
    #[test]
    fn it_sends_no_soft_switch_this_family_would_read_as_prose() {
        let s = qwen36_35b_a3b_antiloop_styletune();
        assert!(matches!(s.chat_format, DialectType::Qwen35));
        assert_eq!(s.dialect.no_think, "");
        assert_eq!(s.dialect.no_think_block, "<think>\n\n</think>\n\n");
    }

    /// **The stock checkpoint is the same family, so the same dialect.** Under
    /// `ChatML` the `/no_think` switch reaches the model as text, every
    /// suppressed turn reasons into its budget, and zend's titler comes back
    /// empty.
    #[test]
    fn the_stock_checkpoint_sends_no_soft_switch_either() {
        let s = qwen36_35b_a3b_q4();
        assert!(matches!(s.chat_format, DialectType::Qwen35));
        assert_eq!(s.dialect.no_think, "");
        assert_eq!(s.dialect.no_think_block, "<think>\n\n</think>\n\n");
    }

    /// **The donor is the stock checkpoint both halves were tuned from.** Anything else would
    /// supply recurrent gates from a different model.
    #[test]
    fn the_gate_donor_is_the_stock_checkpoint() {
        let (repo, rev, file) = quantized_qwen36_moe::QWEN36_35B_A3B;
        assert_eq!(
            qwen36_35b_a3b_antiloop_styletune().gate_donor,
            Some((repo.to_string(), rev.to_string(), file.to_string()))
        );
    }

    /// **`0.7 / 0.95`, `top_k 20`, `presence_penalty 1.5`, reasoning or not** — on the
    /// lineage's steering, which is the arch row's.
    #[test]
    fn it_decodes_at_the_hybrids_numbers_in_both_modes() {
        let s = qwen36_35b_a3b_antiloop_styletune();
        let lineage = SamplingConfig::for_gguf_architecture(ARCH);
        let think_off = s.non_thinking_sampling.clone().expect("a thinking family");
        for c in [&s.default_sampling, &think_off] {
            assert_eq!((c.temperature, c.top_p), (0.7, 0.95));
            assert_eq!(c.presence_penalty, 1.5);
            assert_eq!(c.top_k, lineage.top_k);
        }
        // A turn that declares a mode adopts that mode's row — the same pair either way.
        let declared = s
            .default_sampling
            .clone()
            .with_mode_sampling_for(ThinkMode::Off);
        assert_eq!((declared.temperature, declared.top_p), (0.7, 0.95));
    }

    /// **The cast's boost survives.** `for_character_dialogue` widens the think-off row on top
    /// of the preset, so a character on an ordinary turn decodes at `1.0 / 0.95`.
    #[test]
    fn a_cast_still_decodes_hotter_on_a_think_off_turn() {
        let cast = qwen36_35b_a3b_antiloop_styletune()
            .default_sampling
            .for_character_dialogue()
            .with_think_mode(ThinkMode::Off, 512);
        assert_eq!((cast.temperature, cast.top_p), (1.0, 0.95));
    }
}
