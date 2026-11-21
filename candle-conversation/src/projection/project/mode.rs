//! Names the inference phase a projection is computed for, and the scoring each
//! phase uses.
//!
//! Decode and prefill produce different query vectors, so each mode resolves its
//! own collection scoring. Decode uses a span score on the pragmatic depth, since
//! its queries form a coherent run. Prefill uses a per-token excess score that
//! needs no threshold, because its signal sits below the hit threshold. The
//! figures in the comments are the calibration measured on Qwen3-30B-A3B data.

/// Which inference phase a projection is being computed for.
///
/// Decode and prefill produce structurally different Q vectors and so need
/// different collection-scoring configs (calibrated 2026-05-17 against real
/// Qwen3-30B-A3B data):
///
/// - **Decode** — Q vectors captured while the model *generates* a tool-call
///   response.  They form a coherent run, so `Span{alpha:2.0}` on pragmatic
///   depth dominates (min_ratio 5.96).  This is the steady-state mode used by
///   continuous reprojection during decode.
/// - **Prefill** — Q vectors captured while the model *reads* the user prompt.
///   The signal is weak and lives *below* the hit threshold, so run-based
///   (`Span`) and extreme-value (`Max`) formulas miss it.  `PerTokenExcess`
///   on pragmatic depth — recentered on the noise baseline, per-probe-token,
///   threshold-free — recovers it (full-corpus top-1 58% / top-3 74%, vs
///   `Max` 38% and `Span` 43%).  The intra/inter ratio is still thin, so the
///   `score_threshold` gate is skipped; selection is pure top-k by rank.
///   Used for the initial-guess section injection before decode refines it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionMode {
    Prefill,
    Decode,
}

/// Section-scoring configuration resolved from a [`ProjectionMode`].
pub(super) struct CollectionScoring {
    /// When `false`, `score_threshold` is not used as a gate — every section
    /// competes on rank alone (the prefill ratio is too thin to threshold).
    /// Belief-driven collections ignore this (the policy's `min_score` gates).
    pub(super) apply_threshold: bool,
}

impl ProjectionMode {
    pub(super) fn collection_scoring(self) -> CollectionScoring {
        match self {
            ProjectionMode::Decode => CollectionScoring {
                apply_threshold: true,
            },
            ProjectionMode::Prefill => CollectionScoring {
                apply_threshold: false,
            },
        }
    }
}
