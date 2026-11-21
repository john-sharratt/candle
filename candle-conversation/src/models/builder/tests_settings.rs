//! Tests for how the thinking and selection-budget settings combine.
//!
//! Only an explicit `false` suppresses thinking, so a tenant that sets nothing
//! leaves a shared engine's `true` in place. A selection budget is refused for a
//! model that does not select, before the checkpoint is opened. Every model is built
//! with the default stratified selection unless the caller replaces it whole.

#[cfg(test)]
pub(super) mod thinking_tests {
    use crate::models::Model;

    /// **The core's `thinking(true)` leaves a tenant that asked for nothing
    /// exactly where it was.** A shared core turns thinking on when any tenant
    /// does (`CoreRequests::merge`), so npcd — which sets nothing — runs under
    /// zend's `true`. That is only safe because the switch does one thing:
    /// `false` suppresses thinking for every turn; `true` and unset both leave
    /// each turn's own think mode to decide.
    #[test]
    fn only_an_explicit_false_suppresses_thinking() {
        let unset = Model::Qwen35_0_8B_Q8.builder();
        assert!(!unset.should_suppress_thinking());
        assert!(!unset.thinking(true).should_suppress_thinking());
        assert!(Model::Qwen35_0_8B_Q8
            .builder()
            .thinking(false)
            .should_suppress_thinking());
    }
}

#[cfg(test)]
pub(super) mod selection_budget_tests {
    use candle::Device;

    use crate::models::identity::WeightFiles;
    use crate::models::Model;

    /// **A selection budget is refused where nothing selects.** Accepted, it
    /// would load a model that ignores it and report a dense control that was
    /// never run. The refusal comes before the checkpoint is opened, so the path
    /// is never read.
    #[test]
    fn a_selection_budget_is_refused_for_a_model_without_qsa() {
        let err = Model::Qwen35_0_8B_Q8
            .builder()
            .qsa_selection_budget(Some(1 << 20))
            .load_model(&WeightFiles::single("never-read.gguf"), &Device::Cpu, None)
            .err()
            .expect("a model whose attention does not select must refuse a budget");
        assert_eq!(
            err.to_string(),
            "Qwen35Dense has no QSA selection, so a selection budget of 1048576 position(s) \
             would change nothing"
        );
    }

    /// **Every model is built with the stratified selection unless told
    /// otherwise** — the setting is the engine's, not one architecture's, and a
    /// caller replaces it whole.
    #[test]
    fn every_model_builds_with_the_default_strata_unless_told_otherwise() {
        use candle_transformers::models::selection_strata::{Recent, StrataTokens};
        for model in [Model::Qwen35_0_8B_Q8, Model::Qwen3_30B_A3B_Q4] {
            assert_eq!(
                model.builder().qsa_strata,
                StrataTokens {
                    window: 262_144,
                    recent: 8192,
                    mode: Recent::Candidate,
                    pinned: 0,
                }
            );
        }
        let whole = Model::Qwen35_0_8B_Q8
            .builder()
            .qsa_strata(StrataTokens::WHOLE);
        assert_eq!(whole.qsa_strata, StrataTokens::WHOLE);
    }
}
