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
pub mod gather;
pub mod gpu;
pub mod packed;
pub mod raw_store;
pub mod scan;
pub mod selection;
pub mod wide_sig;

pub use belief::{ToolBelief, DEFAULT_LEAK_BETA};
pub use gather::{belief_step, score_slots, score_slots_weighted, SlotBelief};
pub use gpu::{score_batched_gpu, BatchedGpuGallery, SegmentInput};
pub use packed::{score_packed, PackedGallery};
pub use raw_store::extract_q_vector_r16;
pub use scan::{score_provenance_late_fusion, score_provenance_late_fusion_weighted};
pub use selection::{GroupBudget, SectionPolicy, SectionSelector};
pub use wide_sig::{decode_wide_sigs, encode_wide_sigs, fold_provenance, WideQSig};
