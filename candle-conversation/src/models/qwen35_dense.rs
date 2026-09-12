//! Qwen3.5-9B (dense hybrid) model presets.
//!
//! The lineage's dense member: the same hybrid stack as its routed siblings —
//! gated-DeltaNet layers carrying a recurrent state, attention layers carrying
//! paged K/V — with the mixture taken out.
//!
//! # Why an NPC engine runs the dense one
//!
//! An NPC engine's workload is many small minds thinking concurrently, not one
//! large mind thinking hard. That inverts the usual MoE argument. A routed model
//! amortises expert loads across a wave of sessions stepping through layers
//! together, which is excellent when the sessions are doing similar work — and a
//! hundred characters in different places, on different concerns, reaching
//! different experts, is the case where that amortisation is weakest. The dense
//! model has no expert cache to thrash and no routing to mispredict, so a
//! character's cost is the same whether it is the only one awake or one of a
//! hundred.
//!
//! # One preset, and how a deployment changes it
//!
//! [`Model::Qwen35_9B_Q6`] is the stock instruct model, and it is what `npcd`
//! runs. It carries no adapters: a preset's adapters are downloaded and loaded
//! by every deployment that uses it, so one belongs here only if deployments
//! are meant to run it. Worked examples of the adapter path live where they
//! cost nothing — the gates, and `checkpoints.Qwen35_9B_LoRA` in the override.
//!
//! A deployment wanting a different checkpoint (a fine-tune) or a different
//! adapter does not edit this file: it writes `models.override.yaml` at the
//! workspace root, which is gitignored and replaces both. See
//! [`crate::models::overrides`] for why that indirection exists — a private
//! fine-tune belongs on the machine that runs it, not in a public repository.

use super::{ModelArch, ModelSpec};
use crate::{config::SamplingConfig, models::DialectType};
use candle_transformers::models::quantized_qwen35;

/// Overridden per character at run time — a character's prompt is assembled from
/// its own layers (`npcd/src/engine/prompt.rs`). This is what a bare engine gets
/// with no character attached, which is only ever an operator probing the model.
const PROMPT: &str = "You are a helpful, accurate, and concise assistant.";

/// The GGUF `general.architecture` string this lineage's checkpoints carry.
///
/// Named once because it is the key into [`SamplingConfig::for_gguf_architecture`], and a
/// preset's sampling is whatever that lookup returns — `ModelBuilder::from_spec` copies
/// `default_sampling` verbatim and only `Model::custom` ever re-reads the checkpoint. So a
/// wrong string here is not a wrong label, it is the model decoding on another generation's
/// published numbers, with nothing at load to say so.
const ARCH: &str = "qwen35";

/// Qwen3.5-9B Q6_K — the stock instruct model, 9 B dense.
///
/// # Why the dialect is `Qwen35` and not `ChatML`
///
/// It was `ChatML`, and [`DialectType::Qwen35`] — written for this family,
/// documented at length — had no users at all. The two agree on every turn
/// marker, so the mistake produced a model that loaded, ran, and answered;
/// they differ in exactly one place, and it is one this checkpoint cannot
/// ignore.
///
/// ChatML prepends `/no_think\n` to every user turn as a soft switch.
/// **This checkpoint's chat template contains no such switch** — grepping the
/// shipped GGUF for `no_think` returns nothing — so the marker is not a switch
/// to it, it is the first line of text in every situation a character reads.
/// The family suppresses reasoning by opening the assistant turn with an
/// already-closed think block instead, which `Dialect::qwen35` says by leaving
/// `no_think` empty.
///
/// That also explains a fault recorded as unsolved in `npcd::engine::mind`:
/// the checkpoint opening `<think>` and never closing it, every turn discarded
/// whole as reasoning, and three separate suppression attempts — the `no_think`
/// glue, `thinking_effort`, the `ThinkMode::Off` close budget — all failing.
/// Every one of those is a ChatML mechanism, and none of them is what this
/// family honours. The note there concludes "the remaining fault is below the
/// prompt"; this line is what was below it.
pub(super) fn qwen35_9b_q6() -> ModelSpec {
    let chat_format = DialectType::Qwen35;
    ModelSpec {
        arch: ModelArch::Qwen35Dense,
        loras: Vec::new(),
        dialect: chat_format.dialect(),
        chat_format,
        // The `-MTP-` repo, for the same reason its routed sibling pins one: the
        // plain conversion drops the NextN tensors, and a checkpoint without
        // them cannot speculate. The failure is silent — speculation is
        // lossless, so the answers are identical and only the decode rate
        // falls — which is exactly the kind of regression nobody finds.
        //
        // Not a copy of the gate's coordinates but the same ones: this is the
        // checkpoint `quantized_qwen35`'s C-ladder gate runs against, so the
        // KV threshold row in `QWEN35_9B_KV_FACTORS` was derived on the bytes
        // this serves.
        model_repo: quantized_qwen35::QWEN35_9B.0.into(),
        model_filename: quantized_qwen35::QWEN35_9B.2.into(),
        model_bytes: 7_540_192_896,
        // **The same commit the gate loads.** `QWEN35_9B` has pinned one since an upstream
        // re-upload silently invalidated a threshold tuning; taking `.0` and `.2` and dropping
        // `.1` left the gate on a commit and the daemon on `main`, so the two could load
        // different bytes from one repo name — and the KV factor row measured on the gate's
        // checkpoint would be applied to whatever the daemon happened to fetch.
        model_rev: quantized_qwen35::QWEN35_9B.1.into(),
        gate_donor: None,
        tokenizer_repo: quantized_qwen35::TOKENIZER_REPO.into(),
        tokenizer_rev: quantized_qwen35::TOKENIZER_REV.into(),
        default_system_prompt: PROMPT.into(),
        max_seq_len: 8192,
        // **The lineage's own arch string, not Qwen3's.** A preset's sampling is taken as-is by
        // `ModelBuilder::from_spec` — only `Model::custom` re-detects from the GGUF — so asking
        // for `"qwen3"` here is the whole decision, and it silently ran this model on the
        // previous generation's published numbers while the `qwen35` arm sat unreachable.
        default_sampling: SamplingConfig::for_gguf_architecture(ARCH),
        supports_thinking: true,
        non_thinking_sampling: SamplingConfig::non_thinking_for_gguf_architecture(ARCH),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_transformers::models::dialect::{CallStyle, Dialect};

    /// The stock preset and the gate must name the same bytes. If they drift,
    /// the KV threshold row tuned on the gate's checkpoint is being applied to a
    /// different one — the failure `QWEN35_9B_KV_FACTORS`' own doc comment
    /// describes as invisible until a top-rung session goes wrong.
    #[test]
    fn the_stock_preset_serves_the_checkpoint_the_gate_measures() {
        let s = qwen35_9b_q6();
        assert_eq!(s.model_repo, quantized_qwen35::QWEN35_9B.0);
        assert_eq!(s.model_filename, quantized_qwen35::QWEN35_9B.2);
    }

    /// **The checkpoint carries its speculation head.** A drafter-less
    /// checkpoint answers identically and decodes at half the rate; nothing
    /// reports it, so the repo is pinned and asserted.
    #[test]
    fn the_checkpoint_carries_its_speculation_head() {
        let s = qwen35_9b_q6();
        assert!(
            s.model_repo.contains("-MTP-") || s.model_filename.contains("-MTP-"),
            "{:?} has no MTP marker — it cannot speculate",
            s.model_filename
        );
    }

    /// **The dense arch, not the routed one.**
    ///
    /// The two loaders each refuse the other's checkpoint — correctly, since a
    /// dense file loaded through the routed path would stand up an expert cache
    /// over weights that have none. Declaring `Qwen35Hybrid` made npcd spend
    /// five minutes loading and then fail with `is a dense checkpoint` from a
    /// loader it should never have reached.
    #[test]
    fn it_declares_the_dense_arch_and_its_own_dialect() {
        let s = qwen35_9b_q6();
        assert!(
            matches!(s.arch, ModelArch::Qwen35Dense),
            "arch {:?} routes to the MoE loader, which refuses a dense checkpoint",
            s.arch
        );
        assert!(matches!(s.chat_format, DialectType::Qwen35));
    }

    /// **The switch this family does not have must not be sent to it.**
    ///
    /// This asserted `ChatML`, which is where the bug lived: the two dialects
    /// agree on every marker and differ only here, so the wrong one loads,
    /// runs, and answers — while `/no_think\n` is prepended to every user turn
    /// as literal text, because the checkpoint's own template has no such
    /// switch to consume it. Asserted on the resolved dialect rather than on
    /// the enum, so the thing checked is what the conversation layer will
    /// actually emit.
    #[test]
    fn it_does_not_send_a_soft_switch_this_family_would_read_as_prose() {
        let d = qwen35_9b_q6().dialect;
        assert_eq!(d.no_think, "", "a `/no_think` marker reaches the character");
        assert!(!d.has_no_think_switch());
        // Suppression is the closed block prefilled after the assistant header,
        // which is the mechanism this family actually honours.
        assert_eq!(d.no_think_block, "<think>\n\n</think>\n\n");
        // And the markers are still ChatML's — the families differ in one
        // place, and a change here would be a different bug.
        assert_eq!(d.user_start, Dialect::chat_ml().user_start);
        assert_eq!(d.assistant_start, Dialect::chat_ml().assistant_start);
    }

    /// It writes calls in its own syntax, which is not JSON. See
    /// `candle_transformers::models::dialect::CallStyle`.
    #[test]
    fn it_calls_tools_the_way_its_template_does() {
        let d = qwen35_9b_q6().dialect;
        assert_eq!(d.call_style, CallStyle::FunctionBlock);
        assert!(d.call_style.multi_call());
        assert!(d.call_style.carries_results());
    }

    /// The assembled character prompt is large before any history. A context
    /// that leaves no room for the window makes the gather pointless.
    #[test]
    fn it_leaves_room_for_a_characters_history() {
        assert!(qwen35_9b_q6().max_seq_len >= 8192);
    }

    /// **The preset carries no adapter, and that is the point.**
    ///
    /// It used to carry one — `uknowae/bambara-qwen3.5-9b`, a Bambara language
    /// LoRA — purely so the adapter path had a worked example in the repo. That
    /// cost every deployment of this preset a 112 MB download of an
    /// undocumented, unlicensed, `main`-tracking community adapter that nothing
    /// ever selected: `set_lora` has no callers, so the wave resolved `None` and
    /// ran the base weights every time.
    ///
    /// The machinery it was meant to demonstrate is exercised without it — by
    /// `checkpoints.Qwen35_9B_LoRA` in `models.override.yaml`, by the batched
    /// LoRA gate, and by `resolve_loras`' own tests. A preset is what a
    /// deployment runs, not where an example belongs.
    /// **The preset decodes on its own generation's numbers, not the previous one's.**
    ///
    /// `from_spec` copies `default_sampling` verbatim and only `Model::custom` re-detects from
    /// the GGUF, so the arch string passed here *is* the sampling decision. It read `"qwen3"`
    /// while the table's `qwen35` arm sat unreachable, and nothing failed: the steering fields
    /// are shared between the two arms, so the model still terminated correctly and only the
    /// four sampled numbers were the older generation's. Comparing against the table directly
    /// is what makes that visible.
    #[test]
    fn the_preset_samples_as_its_own_architecture() {
        let got = qwen35_9b_q6().default_sampling;
        let want = SamplingConfig::for_gguf_architecture("qwen35");
        let previous = SamplingConfig::for_gguf_architecture("qwen3");

        // The four numbers the two generations publish differently. `SamplingConfig` has no
        // `PartialEq`, and naming them is clearer regardless: these are the values Qwen
        // documents for this lineage, and the ones an earlier version of this preset was not
        // using.
        assert_eq!((got.temperature, got.top_k), (want.temperature, want.top_k));
        assert_eq!(
            (got.top_p, got.presence_penalty),
            (want.top_p, want.presence_penalty)
        );
        assert_ne!(
            (got.top_k, got.presence_penalty),
            (previous.top_k, previous.presence_penalty),
            "qwen3 and qwen35 publish different sampling; if these match, the table has \
             collapsed the two rows and this preset can no longer tell which it got"
        );
        // The steering is shared between the arms, which is exactly why the wrong arch string
        // produced no visible failure — assert it survived the change.
        assert!(got.force_segment_close_after > 0 && got.forced_eos_after > 0);
    }

    #[test]
    fn the_preset_carries_no_adapter() {
        assert!(
            qwen35_9b_q6().loras.is_empty(),
            "a preset's adapters are downloaded and loaded by every deployment that \
             uses it, so one may only be here if deployments are meant to run it"
        );
    }

    /// Every adapter name a preset carries must be distinct.
    ///
    /// The name is the key a conversation selects by and the key the model's
    /// registry stores under, so a duplicate does not collide loudly — the
    /// second load silently replaces the first, and conversations asking for
    /// one get the other.
    #[test]
    fn adapter_names_within_a_preset_are_unique() {
        let s = qwen35_9b_q6();
        let mut seen = std::collections::HashSet::new();
        for l in &s.loras {
            assert!(
                seen.insert(l.name.clone()),
                "duplicate adapter name {:?}",
                l.name
            );
        }
    }
}
