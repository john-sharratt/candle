//! Attentional provenance retrieval subsystem.
//!
//! Provides binary directional signatures for each token's Q vector (captured
//! in R16 KV format) and a mmap-backed file that stores and scans them.
//!
//! # Pipeline
//!
//! ## Index side (automatic, per turn seal)
//!
//! The R16 KV format co-stores raw F16 Q vectors alongside K values.  Inside
//! the post-Done seal step, the scheduler extracts Q sign-bits at three layer depths
//! (syntactic ~15%, semantic ~50%, pragmatic ~85%), appends one chunk-group
//! triplet per 32-token block to the shared `ProvenanceFile`, and stashes the
//! new `SigEntry` values on the workspace substrate's per-`(group, turn)` entry.
//!
//! ## Query side (continuous, in-decode)
//!
//! The scheduler's reprojection path extracts live Q sign-bits from
//! the active decode at every reprojection cadence trigger, runs a
//! BDP scan against the per-turn `SigEntry` directory in the
//! workspace `Conversation` (substrate), updates per-turn scores,
//! and re-projects the visible window.  No fork, no separate query
//! API — the live sequence is its own probe.

pub mod raw_store;
pub mod scan;
pub mod signature;
pub mod store;

pub use raw_store::{
    band_layer_indices, build_token_blob, extract_k_vector, extract_q_vector_r16,
    RawFileHeader, RawProvenanceFile, RawSigEntry,
};
pub use scan::{BdpScanner, TokenHit, DEFAULT_HIT_THRESHOLD, DEFAULT_SPAN_ALPHA, DEFAULT_TOP_K};
pub use signature::{
    extract_mh_signatures_from_r16_dump, extract_signatures_from_r16_dump,
    merge_turn_signatures_xor, r16_block_to_turn_signatures, r16_block_to_turn_signatures_mh,
    TokenSignature, TurnSignatures,
};
pub use store::{ProbeSignatures, ProvenanceFile, SigEntry, TurnChunkRank};
