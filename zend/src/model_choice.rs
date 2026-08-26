//! The single place zend decides which model it runs.
//!
//! Everything downstream — the downloader's repo/filename/size coordinates and
//! the session's model builder — derives from [`model`], so the choice cannot
//! drift between call sites.
//!
//! Zen Code runs **Qwen3.6-35B-A3B**, the hybrid: 40 layers at 3:1, so 30
//! gated-DeltaNet layers carrying a recurrent state and 10 attention layers
//! carrying paged K/V. Three quarters of the stack has no K/V to splice, which
//! is what the recurrent-state persistence work in
//! `docs/deltanet_state_persistence.md` exists to support.
//!
//! There is no VRAM ladder here. The lineage ships one quant (`UD-Q4_K_M`), and
//! a MoE model's resident footprint is its dense weights plus whatever expert
//! working set fits — the three-tier expert cache pages the rest — so parameter
//! count does not decide whether a card can run it. A bigger card buys speed,
//! not feasibility.

use candle_conversation::models::Model;

/// The model zend runs.
pub fn model() -> Model {
    Model::Qwen36_35B_A3B_Q4
}

#[cfg(test)]
mod tests {
    use super::model;
    use candle_conversation::models::{Model, ModelArch};

    #[test]
    fn zend_runs_the_hybrid() {
        assert!(matches!(model(), Model::Qwen36_35B_A3B_Q4));
        assert!(matches!(model().spec().arch, ModelArch::Qwen35Hybrid));
    }

    /// The plain `-GGUF` conversion drops the NextN tensors, and a checkpoint
    /// without them cannot speculate. Nothing downstream would notice:
    /// speculation is lossless, so the answers are identical and only the
    /// decode rate falls. Pin the repo that carries the head.
    #[test]
    fn the_checkpoint_carries_the_speculative_head() {
        let spec = model().spec();
        assert!(
            spec.model_repo.contains("-MTP-"),
            "zend's checkpoint must come from the -MTP- repo, got {:?}",
            spec.model_repo,
        );
    }
}
