//! Cooperative ingest cancellation — an engine-layer concern.
//!
//! The startup ingest, the background reconcile, and uploads all drive the same
//! engine (scheduler + persistence). A graceful shutdown mid-ingest must stop
//! that ingest at a safe boundary so the persistence thread can drain the tier
//! tail hot→warm→cold instead of losing it. This flag is the signal.
//!
//! It lives HERE (not in the app/daemon) for two reasons: the ingest and the
//! summary-decode wait it needs to interrupt are engine internals
//! ([`crate::Sequence::ingest_scope_roundtrip_indices`]), and the flag must be
//! reachable while the engine is still being constructed on the loader thread —
//! before any [`ConversationEngine`](crate::ConversationEngine) handle is
//! published — which a free function (no engine reference) provides.
//!
//! **Scope & lifecycle.** The flag is process-scoped (one latch for the whole
//! process, not per-engine), so a caller that runs more than one engine — or the
//! test binary, which runs many — must treat it as shared. It is **resettable**:
//! [`reset_ingest_cancel`] clears it at the start of a load so a fresh load (or
//! the next test) never inherits a prior run's cancel, and an embedder can
//! re-arm ingest after a cancelled one.

use std::sync::atomic::{AtomicBool, Ordering};

static INGEST_CANCEL: AtomicBool = AtomicBool::new(false);

/// Ask every in-flight ingest to stop at its next file / cluster / scope-chunk
/// boundary (and the in-flight summary decode to abandon its wait). Idempotent.
pub fn request_ingest_cancel() {
    INGEST_CANCEL.store(true, Ordering::SeqCst);
}

/// Clear the cancel latch so a new load starts un-cancelled. Called at the start
/// of a load (and usable by tests / embedders to re-arm ingest after a prior
/// cancel), since the flag is process-scoped and would otherwise persist across
/// loads within one process.
pub fn reset_ingest_cancel() {
    INGEST_CANCEL.store(false, Ordering::SeqCst);
}

/// Whether a shutdown has asked ingest to stop. Polled in the ingest item loops
/// and the interruptible decode-wait. A `Relaxed` load is sufficient — this is a
/// one-way latch (until [`reset_ingest_cancel`]), not a synchronisation point.
pub fn ingest_cancelled() -> bool {
    INGEST_CANCEL.load(Ordering::Relaxed)
}
