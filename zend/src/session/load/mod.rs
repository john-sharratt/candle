//! Builds the daemon's inference state from the workspace, in the order its startup
//! steps depend on.
//!
//! The schema is installed and final before any conversation is built or any
//! ingest runs. The thinking-block token ids are resolved after the engine exists,
//! because every think-block budget is gated on them. A stopped load is a stop and
//! not a failure: `load_stopped` drains the engine on the loader thread that owns
//! it, and what was sealed stays durable while half-built work is found and redone
//! at the next start.
//!
//! - `startup`: the ordered load steps.

use super::*;

/// Whether a stop has ended the load at the boundary after `step` — and if
/// so, the engine drained here, on the loader thread that owns it, exactly as
/// the startup-ingest cancel drains it. The caller then returns `Ok(None)`: a
/// stopped load is a stop, not a failure, and what each step sealed so far is
/// durable while what it left half-built is found and redone next start (a
/// calibration case un-archived, an ingest unit uncommitted).
pub(super) fn load_stopped(engine: &ConversationEngine, step: &str) -> bool {
    if !crate::ingest_cancel::cancelled() {
        return false;
    }
    tracing::warn!("shutdown after {step} — draining the engine on the loader thread");
    if let Err(e) = engine.shutdown() {
        tracing::error!("shutdown after {step}: drain failed: {e}");
    }
    true
}

mod startup;
