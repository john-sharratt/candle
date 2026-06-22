//! The §6 summary probe — slot recipe + runner abstraction.
//!
//! A "summary probe" is the §6 forward-continuation that produces a
//! compressed turn over a set of children:
//!
//! ```text
//!     ┌─────────────────────────────────────────────────────────┐
//!     │  "compressor" system prompt + the children turns' text   │
//!     ├─────────────────────────────────────────────────────────┤
//!     │  Prefill:  "Compress the conversation above…"            │
//!     │  Decode:   a faithful, much shorter rewrite of the turns │
//!     └─────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//!     A new substrate turn is sealed with `kind = SummaryOfTurns`
//!     (or `SummaryOfSummaries`), carrying the decoded compressed
//!     turn's K/V (so a later projection can inject it).
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

/// One probe request — "compress these turns into a new turn of this
/// `kind`".  The summariser thread emits one of these per pending-turn
/// absorption and one per dirty-set sweep regeneration.
#[derive(Debug, Clone)]
pub struct ProbeRequest {
    pub timeline: TimelineId,
    /// What the produced compressed turn's tree role will be.  The
    /// runner doesn't strictly need this — the summariser sets the
    /// `TreeNodeMeta` after sealing — but it's useful for tracing and
    /// for runners that choose different system prompts per kind.
    pub kind: TurnKind,
    /// The structural tree children this node records, in chronological
    /// order, *and* the turns the compressor reads to produce its text. For
    /// `SummaryOfTurns`: the single Normal turn. For `SummaryOfSummaries`:
    /// a run of `MERGE_FANOUT` same-level summary turns. Written into the node's
    /// `TreeNodeMeta`; the compression pass injects each child's two halves.
    pub children: Vec<TurnIndex>,
    /// The node's tree height (`SummaryOfTurns` leaves = 1; a
    /// `SummaryOfSummaries` = its children's level + 1). Drives the structural
    /// roll-up's height-based directory-depth pruning.
    pub height: u8,
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

    /// Execute several probes with as much concurrency as the runner
    /// supports, returning results in request order.  The scheduler-backed
    /// runner submits them all up front so their decodes batch together in
    /// the wave loop; the default is a serial fallback for runners (mock)
    /// that have no concurrency.
    fn run_batch(&self, requests: Vec<ProbeRequest>) -> Vec<Result<ProbeResponse, ProbeError>> {
        requests.into_iter().map(|r| self.run(r)).collect()
    }
}

// The compressor system prompt, the per-half instruction, and the token/word
// budget are no longer hardcoded here — each projection layer (and section
// group) declares its own `summary` block (system prompt + user prompt +
// max_tokens) in the template; the scheduler reads the target layer's summary
// when it runs a compression pass.

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
}
