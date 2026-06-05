//! The §6 summary probe — slot recipe + runner abstraction.
//!
//! A "summary probe" is the §6 forward-continuation that produces a
//! summary turn over a set of children K/V chunks:
//!
//! ```text
//!     ┌─────────────────────────────────────────────────────────┐
//!     │  Synthetic "summariser" system section (pinned)          │
//!     ├─────────────────────────────────────────────────────────┤
//!     │  inject_sealed_at_tail: child K/V chunks                 │
//!     ├─────────────────────────────────────────────────────────┤
//!     │  Prefill:  "Summarise the above turns."                  │
//!     │  Decode:   structured-JSON summary content               │
//!     └─────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//!     A new substrate turn is sealed with `kind = SummaryOfTurns`
//!     (or `SummaryOfSummaries`), carrying the decoded content + the
//!     Q sign-bits captured during decode.
//! ```
//!
//! The probe execution itself happens on the GPU via the scheduler.
//! This module defines the abstraction:
//!
//! - [`ProbeRequest`] — what the summariser asks for.
//! - [`ProbeResponse`] — the sealed substrate `TurnIndex` of the new
//!   summary turn.
//! - [`ProbeError`] — soft (retry-on-next-pass) and hard (stop the
//!   thread) variants.
//! - [`ProbeRunner`] — the trait the summariser thread holds.  Two
//!   production-quality impls live in `summariser.rs`:
//!     1. [`ChannelProbeRunner`](super::summariser::ChannelProbeRunner)
//!        — sends a `SchedulerRequest::SubmitSummaryProbe` and blocks
//!        for the response.
//!     2. [`MockProbeRunner`](super::summariser::MockProbeRunner) —
//!        appends a placeholder summary turn directly to the
//!        substrate, used by unit tests that exercise the tree
//!        bookkeeping without a model.

use crate::projection::{TimelineId, TurnIndex};
use crate::summary_tree::TurnKind;

/// One probe request — "summarise these children into a new turn of
/// this `kind`".  The summariser thread emits one of these per
/// pending-turn absorption and one per dirty-set sweep regeneration.
#[derive(Debug, Clone)]
pub struct ProbeRequest {
    pub timeline: TimelineId,
    /// What the produced summary turn's tree role will be.  The
    /// runner doesn't strictly need this — the summariser sets the
    /// `TreeNodeMeta` after sealing — but it's useful for tracing and
    /// for runners that choose different system prompts per kind.
    pub kind: TurnKind,
    /// Children to summarise, in chronological order.  For
    /// `SummaryOfTurns`: the run of Normal turns.  For
    /// `SummaryOfSummaries`: exactly two summary turns (`[left,
    /// right]`).
    pub children: Vec<TurnIndex>,
}

/// One probe response — the substrate-level `TurnIndex` of the
/// freshly-sealed summary turn.  The summariser writes the tree
/// metadata (`kind`, `children`, `tree_height`, `dirty=false`) onto
/// that index after receiving this.
#[derive(Debug, Clone, Copy)]
pub struct ProbeResponse {
    pub sealed_turn: TurnIndex,
}

/// Failure mode for a probe run.
#[derive(Debug, Clone)]
pub enum ProbeError {
    /// The runner couldn't produce a valid summary for *this*
    /// request, but the substrate is still in a sane state.  The
    /// summariser re-enqueues the failed children and tries again on
    /// the next pass.  Typical causes: model output wasn't valid
    /// JSON, GPU contention timeout, transient I/O failure.
    Soft(String),
    /// Unrecoverable failure — GPU error, scheduler shutdown, etc.
    /// The summariser thread logs and stops.  Engine teardown
    /// proceeds normally.
    Hard(String),
}

impl std::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProbeError::Soft(msg) => write!(f, "soft probe error: {msg}"),
            ProbeError::Hard(msg) => write!(f, "hard probe error: {msg}"),
        }
    }
}

impl std::error::Error for ProbeError {}

/// The summariser's probe interface.  One trait, two production
/// impls (channel-backed for live operation, mock for unit tests).
pub trait ProbeRunner: Send + Sync + 'static {
    /// Execute one probe.  Production runners send a scheduler RPC and
    /// block for the response; mock runners append a placeholder turn
    /// directly to the substrate.  Either way, the returned
    /// `sealed_turn` MUST refer to a substrate turn the summariser
    /// can immediately attach tree metadata to.
    fn run(&self, request: ProbeRequest) -> Result<ProbeResponse, ProbeError>;
}

/// Standard "summariser" system prompt text — the §6.1 recipe's
/// `①` block.  Tokenised once at engine init and pinned as a
/// substrate section so every probe replays it as a cached prefix
/// (zero per-probe cost).
///
/// The prompt mandates exact JSON output; the structured-output
/// grammar enforces it on the sampling side.  Kept short so the
/// pinned section's K/V is cheap.
pub const SUMMARISER_SYSTEM_PROMPT: &str =
    "You are a summariser.  Read the turns above and produce a one-line digest.  \
     Output JSON only, in exactly one of these shapes:\n\
     {\"coherent\": true, \"summary\": \"<one-line digest of the turns>\"}\n\
     {\"coherent\": false, \"split_at\": <index of the first turn of the new topic>}\n\
     Default to coherent=true unless the turns clearly span unrelated topics.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_error_display_includes_kind_and_message() {
        let soft = ProbeError::Soft("bad json".into());
        assert!(soft.to_string().contains("soft"));
        assert!(soft.to_string().contains("bad json"));
        let hard = ProbeError::Hard("gpu oom".into());
        assert!(hard.to_string().contains("hard"));
        assert!(hard.to_string().contains("gpu oom"));
    }

    #[test]
    fn summariser_prompt_demands_json() {
        assert!(SUMMARISER_SYSTEM_PROMPT.contains("JSON"));
        assert!(SUMMARISER_SYSTEM_PROMPT.contains("coherent"));
        assert!(SUMMARISER_SYSTEM_PROMPT.contains("split_at"));
    }
}
