//! The single place zend decides which model it runs.
//!
//! Everything downstream — the downloader's repo/filename/size coordinates and
//! the session's model builder — derives from [`model`], so the choice cannot
//! drift between call sites.
//!
//! Zen Code runs **Qwen3.8-Flash-Next**: 48 layers at 3:1, so 36 gated-DeltaNet
//! layers carrying a recurrent state and 12 attention layers carrying paged K/V,
//! 250 B total at ~13 B active, with a native 262,144-token context.
//!
//! Three quarters of the stack has no K/V to splice, which is what the
//! recurrent-state persistence work in `docs/deltanet_state_persistence.md`
//! exists to support — and this lineage adds two more carried classes on top,
//! a PLE window and a QSA index, neither of which is a delta-rule matrix and
//! neither of which can be rebuilt from restored K/V. Both ride the turn
//! record's model-opaque blob (`docs/qwen38_index_persistence.md`).
//!
//! There is no VRAM ladder here. A MoE model's resident footprint is its dense
//! weights plus whatever expert working set fits — the three-tier expert cache
//! pages the rest — so parameter count does not decide whether a card can run
//! it. A bigger card buys speed, not feasibility.
//!
//! **The engine GGUF is prepared locally, not downloaded.** Nothing publishes
//! it: it is the pinned Q8_0 split with the W4A16 release's experts imported to
//! Q4_KO and the MTP head folded in, merged into one file. `download.rs` reports
//! the build step rather than asking the hub for a name nobody published.

use candle_conversation::models::Model;

/// The model zend runs.
pub fn model() -> Model {
    Model::Qwen38_FlashNext_Q4KO
}

#[cfg(test)]
mod tests {
    use super::model;
    use candle_conversation::models::{Model, ModelArch};

    #[test]
    fn zend_runs_flash_next() {
        assert!(matches!(model(), Model::Qwen38_FlashNext_Q4KO));
        assert!(matches!(model().spec().arch, ModelArch::Qwen4Exp));
    }

    /// **The engine artifact is built here, so resolution must not go asking.**
    ///
    /// Nothing publishes the merged GGUF under its name. A spec that left this
    /// unset would send the daemon to the hub for a 404 on every start, and the
    /// operator would read "download failed" for a file that was never
    /// downloadable — instead of "run the prepare step".
    #[test]
    fn the_engine_artifact_is_prepared_not_downloaded() {
        assert!(model().spec().prepared_from_source);
    }

    /// This lineage suppresses thinking by opening the assistant turn with an
    /// already-closed think block; Qwen3's `/no_think` marker does nothing on
    /// it. Reading a Qwen3 dialect here would leave reasoning blocks in every
    /// reply with nothing failing.
    #[test]
    fn the_dialect_is_the_flash_next_one() {
        assert!(matches!(
            model().spec().chat_format,
            candle_conversation::models::DialectType::Qwen35
        ));
    }

    /// The context the checkpoint actually backs — measured flat from 32K to
    /// 128K, not a declared number. A conversation is bounded by this.
    #[test]
    fn the_native_context_is_not_truncated_by_the_spec() {
        assert_eq!(model().spec().max_seq_len, 262_144);
    }

    /// **The residual stream is BF16, and the session has to be told.**
    ///
    /// Left unset, the session derives its activation width from the KV storage
    /// format, which is F16 for the quantized backing this engine always uses —
    /// a statement about the arena, not about the residual stream. This lineage
    /// publishes `"dtype": "bfloat16"`, and the two are not interchangeable:
    /// both are 16 bits, but F16 stops at 65504 where BF16 carries F32's
    /// exponent, so a block that legitimately produces 1e5 silently becomes
    /// `inf` and the next layer turns it into a NaN.
    #[test]
    fn the_activation_width_is_bf16() {
        assert_eq!(
            model().spec().arch.native_activation_dtype(),
            Some(candle::DType::BF16),
        );
    }
}
