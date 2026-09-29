//! One row per model: what the probe runs, and what it demands of the result.
//!
//! **This file is the surface for adding a model.** Everything else in
//! `fragmentation_probe` is model-agnostic; a new model is a [`ModelProfile`] here
//! plus a test case naming it.
//!
//! # Why the thresholds are per model and not constants
//!
//! The probe's two VRAM gates are ratios over *region* counts, and a region holds
//! whatever fits its size class. Which rung of the size-class ladder a model's chunks
//! land on is a function of `n_kv_head × head_dim / N_PALETTE` and of the compression
//! the policy selects — so two models churning the same number of conversations
//! produce different arena populations, and a burst strands a different number of
//! them. The correctness gate varies for a plainer reason: reproducing a page of prose
//! verbatim is a capability, and a small model does not have it.
//!
//! So a new row's numbers are **measured, not guessed**: run the probe against the
//! model, read the `worst sustained` row of its results table across a few runs, and
//! set the threshold at what a working compaction actually holds. A row copied from
//! another model tests that model's geometry, not this one's.

use crate::models::Model;

/// How a phase-B reply is checked for cross-session contamination.
///
/// **There is no "skip" variant, deliberately.** A probe run with no correctness
/// check would let a compaction that relocates a chunk and leaves one band pointer
/// stale pass every gate it has: the wrong KV does not fault, it answers wrongly, and
/// throughput and geometry both look perfect while it does. The weaker variant below
/// is the floor, not an opt-out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoryGate {
    /// The reply must be a prefix of the expected rewrite, normalised, within the
    /// forward gate's own 5-character tolerance.
    ///
    /// The strongest check available, and the one to prefer: it catches a wrong name,
    /// wrong content, and broken attention alike. It needs a model that can reproduce
    /// a page of prose exactly under greedy decoding, which in practice means the
    /// mid-size class and up.
    Verbatim,
    /// The reply must name its **own** protagonist.
    ///
    /// For a model that cannot reproduce text exactly. It still detects the failure
    /// this check exists for — a session reading another session's KV renames to
    /// *that* session's protagonist, so its own name never appears — while not
    /// depending on fidelity the model does not have.
    ///
    /// Deliberately not "must not contain another session's name": the fixture's names
    /// run from two characters (`Bo`) to twenty, and a two-character name is a
    /// substring of ordinary words, so the negative form reports contamination that is
    /// not there.
    OwnName,
}

/// Everything about a probe run that is a property of the model rather than of the
/// question being asked.
#[derive(Clone, Debug)]
pub struct ModelProfile {
    /// The name a caller selects this row by — the `--model` value, and the test
    /// case's argument.
    pub name: &'static str,
    pub model: Model,
    /// Context the engine is sized for. Bounds how much KV one conversation can hold,
    /// so it is what decides whether the churn can reach the frontier at all.
    pub max_seq_len: usize,
    /// Conversations alive at once through the churn.
    pub concurrency: usize,
    /// Ceiling while saturating, and the number of worker threads spawned.
    pub max_concurrency: usize,
    /// Long-lived conversations created once and held for the whole run — the
    /// immovable neighbours.
    ///
    /// **This is the half that makes holes.** Uniform churn does not fragment: every
    /// arena empties eventually and the pool re-packs from the bottom. Fragmentation
    /// needs frees landing *around things that never move*, which in the daemon is the
    /// priming chain, the dialogue base conversations and the section ingests sitting
    /// resident while ingest units are evicted beneath them.
    pub pinned: usize,
    /// Concurrent sequences in the phase-B comparison batch.
    pub batch: usize,
    /// Minimum sustained VRAM efficiency, as a percentage, below which the run fails.
    ///
    /// Efficiency is `packed_arenas / frontier`: of the ground denied to the weight
    /// side, how much is actually holding KV. The frontier is the denominator because
    /// the frontier is what the weight side loses — the wave transient tier stands
    /// above the highest live arena and `weight_floor` is measured from there, so a
    /// region below the frontier costs the weight side whether it is live, sparse or
    /// free.
    ///
    /// **This threshold is the specification for the compaction pass.** It says how
    /// much VRAM fragmentation is costing, in the one unit that matters, and the same
    /// number passing is what says compaction worked. A run that cannot fail cannot
    /// tell you that.
    pub min_efficiency: usize,
    /// Of the KV ground released during the drain, the minimum percentage the **weight
    /// side must take**, below which the run fails.
    ///
    /// **Freeing ground is only half the job.** The span is
    /// `| persist | KV regions | tier | expert weights |` with `weight_floor` between
    /// the last two, and lowering the arena frontier merely makes it *possible* for
    /// that floor to move left. Something has to actually move it. If it does not,
    /// compaction hands back regions nobody claims and decode is exactly as slow as
    /// before — the work would be invisible in every metric except the one that
    /// matters.
    ///
    /// Vacuously satisfied on a card that already holds every expert slot the model
    /// has: there is then no residency for released ground to buy. The probe reads that
    /// from the engine's own growth ledger rather than inferring it, and says so in the
    /// result.
    pub min_weight_uptake: usize,
    pub story: StoryGate,
}

/// Every model the probe has measured thresholds for.
///
/// Rows are ordered smallest first, which is also the order to bring a new one up in:
/// a threshold that holds on a small model and fails on a large one is usually the
/// workload not reaching the frontier, not the compaction.
pub fn profiles() -> Vec<ModelProfile> {
    vec![
        // The reference row. Q4_K_M on the 30B-A3B is the forward gate's own
        // checkpoint, so this profile's phase-B figures are comparable to that gate's
        // clean rows rather than to another model's.
        //
        // Measured over six runs on a 72 GiB card: steady-state efficiency 94–99%, and
        // the worst figure that persisted across two publishes never below 90. The
        // weight gate reports at-limit here — the whole 19 GiB checkpoint is resident,
        // so there is no residency for released ground to buy — which the probe
        // recognises from the growth ledger.
        ModelProfile {
            name: "qwen3-30b-a3b-q4",
            model: Model::Qwen3_30B_A3B_Q4,
            max_seq_len: 8192,
            concurrency: 24,
            max_concurrency: 40,
            pinned: 6,
            batch: 20,
            min_efficiency: 90,
            min_weight_uptake: 50,
            story: StoryGate::Verbatim,
        },
        // Qwen3.8-Flash-Next. **The row that matters for correctness, not throughput.**
        //
        // This is the arch that carries per-sequence state *outside* the paged K/V — the
        // DeltaNet recurrence, the PLE convolution tail, the QSA index cache — which makes
        // it the only model in the table that can show a compaction or a boundary move
        // corrupting something the K/V sweep never touches. On a live daemon it did: 90
        // seconds after two compaction passes, 15 of 36 layers of persistent recurrent
        // state went non-finite, the logits row came back all-NaN, and the turn truncated
        // on a forced EOS. The 30B row above cannot catch that at all, because it carries
        // no such state.
        //
        // **The VRAM thresholds here are provisional and not yet measured**, unlike the
        // row above. They are set to the same figures so a regression in packing still
        // fails, but the number this row is trusted for today is `story`: 56 GiB of
        // resident experts leaves far less KV room than the 30B has, so the arena
        // population — and therefore what a burst strands — has not been characterised.
        // Measure them before reading an efficiency failure here as a compaction defect.
        //
        // `model` names the widest rung; a caller runs the preset its card's rung
        // loads (`Model::qwen38_flash_next_for`), because every machine holds only
        // its own rung's artifact.
        ModelProfile {
            name: "qwen38-flash-next",
            model: Model::Qwen38_FlashNext_Q4KO,
            max_seq_len: 8192,
            // Narrower than the 30B's, and deliberately: the resident expert grid is 56
            // GiB of a 72 GiB card, so the KV side has a fraction of the span the 30B
            // enjoys and a 24-wide churn cannot be admitted at all.
            concurrency: 8,
            max_concurrency: 16,
            pinned: 4,
            batch: 8,
            min_efficiency: 90,
            min_weight_uptake: 50,
            story: StoryGate::Verbatim,
        },
    ]
}

/// The row a caller named, or `None`.
pub fn profile(name: &str) -> Option<ModelProfile> {
    profiles().into_iter().find(|p| p.name == name)
}

/// Every selectable name, for a driver's usage text and for an error message that
/// tells the reader what they could have said instead.
pub fn names() -> Vec<&'static str> {
    profiles().into_iter().map(|p| p.name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A name that selects nothing is a typo the caller must hear about, and two rows
    /// sharing a name would make one of them unreachable.
    #[test]
    fn every_row_is_selectable_by_a_unique_name() {
        let all = profiles();
        assert!(!all.is_empty(), "a probe with no models cannot run");
        for p in &all {
            assert!(
                profile(p.name).is_some(),
                "{} is in the table but does not resolve",
                p.name,
            );
        }
        let mut seen: Vec<&str> = names();
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        assert_eq!(before, seen.len(), "two rows share a name");
        assert!(profile("no-such-model").is_none());
    }

    /// Thresholds are percentages, and a workload that admits nothing measures
    /// nothing. Cheap to state and it catches a mistyped row before a model load.
    #[test]
    fn every_row_is_internally_coherent() {
        for p in profiles() {
            assert!(
                p.min_efficiency > 0 && p.min_efficiency <= 100,
                "{}: efficiency threshold is a percentage",
                p.name,
            );
            assert!(
                p.min_weight_uptake <= 100,
                "{}: uptake threshold is a percentage",
                p.name,
            );
            assert!(
                p.concurrency > 0 && p.concurrency <= p.max_concurrency,
                "{}: concurrency must be positive and within its own ceiling",
                p.name,
            );
            assert!(p.batch > 0, "{}: phase B needs sequences", p.name);
            assert!(
                p.max_seq_len >= 1024,
                "{}: too little context for a conversation to span arenas",
                p.name,
            );
        }
    }
}
