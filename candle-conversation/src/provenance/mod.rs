//! Attentional provenance retrieval subsystem.
//!
//! Captures each token's `sign(Q)` from the R16 KV format as a compact folded
//! [`WideQSig`], and scores a live decode window against a gallery of past turns'
//! signatures to drive belief-based section/tool selection.
//!
//! # Pipeline
//!
//! ## Index side (automatic, per turn seal)
//!
//! At the post-Done seal step the scheduler gathers each real token's `sign(Q)`
//! across all heads/layers, [`fold_provenance`]-folds it to the locked 1536-bit
//! [`WideQSig`], and stores the per-turn window on the substrate
//! (`wide_q_sigs`). See `docs/tool_selection_provenance_results.md` §23.
//!
//! ## Query side (continuous, in-decode)
//!
//! Each reprojection gathers the live decode window's folded signatures as a
//! probe, scans it against the tag-scoped gallery of past turns
//! ([`crate::projection::Conversation::belief_gallery`] +
//! [`score_slots`]), and updates the online belief
//! ([`belief_step`]) that drives selection — the live sequence is its own probe.

pub mod belief;
pub mod fusion;
pub mod gallery_arena;
pub mod gather;
pub mod gpu;
pub mod packed;
pub mod raw_store;
pub mod scan;
pub mod selection;
pub mod wide_sig;

pub use belief::{ToolBelief, DEFAULT_LEAK_BETA};
pub use fusion::FusionMode;
pub use gallery_arena::GalleryArena;
pub use gather::{
    belief_step, score_slots, score_slots_fused, score_slots_grouped, score_slots_weighted,
    SlotBelief,
};
pub use gpu::{score_batched_gpu, BatchedGpuGallery, SegmentInput};
pub use packed::{score_packed, PackedGallery};
pub use raw_store::extract_q_vector_r16;
pub use scan::{
    heads_per_group, score_provenance_late_fusion, score_provenance_late_fusion_fused,
    score_provenance_late_fusion_grouped, score_provenance_late_fusion_weighted, FOLD_GROUPS,
};
pub use selection::{GroupBudget, SectionPolicy, SectionSelector};
pub use wide_sig::{
    active_fold, decode_wide_sigs, decode_wide_sigs_for_scoring, encode_wide_sigs,
    encode_wide_sigs_with, fold_fits, fold_provenance, fold_provenance_checked,
    fold_provenance_fitted, set_active_fold, wide_sig_fold_params, FoldParams, WideQSig,
};
