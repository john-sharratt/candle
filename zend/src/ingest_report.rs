//! Ingest-completeness reporting, shared by every per-unit ingest pass.
//!
//! A unit (directory / file) whose ingest fails keeps its **prior generation
//! live**, so the
//! workspace stays queryable — but that also means a failed pass looks like a
//! successful one from the outside. Before this module the only trace was a
//! `WARN` per unit plus a count buried in an `INFO` line that said "complete";
//! a run could lose a quarter of its map and still report ready.
//!
//! Completeness is therefore recorded as first-class state: every failure is
//! captured with the directory and the full error chain, published
//! process-globally when the pass ends, and served at `GET /v1/repo_map`. A
//! stale map is now something you can *query*, not something you infer from log
//! archaeology.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};

use serde::Serialize;

/// Failure detail kept per unit. The `error` is the full `{:#}` chain, so
/// the root cause (e.g. a KV VRAM budget refusal) survives to the API without
/// needing the log.
#[derive(Debug, Clone, Serialize)]
pub struct UnitFailure {
    /// Directory or file path, whichever the pass ingests.
    pub unit: String,
    pub error: String,
}

/// How many individual failures the report retains. The count is exact; the
/// detail list is capped so a total wipe-out can't produce a 300-entry payload.
const MAX_RETAINED: usize = 32;

/// Outcome of one per-directory ingest pass.
#[derive(Debug, Clone, Serialize)]
pub struct IngestReport {
    /// Units the pass was asked to ingest.
    pub n_units: usize,
    /// Units that failed (exact, even when `failures` is truncated).
    pub n_failed: usize,
    /// Whether the failure cap tripped and the pass stopped early — the rest of
    /// `n_units` was never attempted, so the map is stale beyond `n_failed` too.
    pub aborted: bool,
    /// Per-unit detail, capped at [`MAX_RETAINED`].
    pub failures: Vec<UnitFailure>,
}

impl IngestReport {
    /// Whether the repo map is known to be incomplete — the single predicate
    /// callers should branch on rather than re-deriving it from counts.
    pub fn is_incomplete(&self) -> bool {
        self.aborted || self.n_failed > 0
    }
}

/// Live collector shared by the pass's workers.
///
/// Replaces the bare `AtomicUsize` the pass used to carry: the count alone
/// could not answer "which directories are stale, and why", which is exactly
/// what a partial map needs to report.
pub struct Failures {
    n: AtomicUsize,
    abort: AtomicBool,
    detail: Mutex<Vec<UnitFailure>>,
}

impl Default for Failures {
    fn default() -> Self {
        Self::new()
    }
}

impl Failures {
    pub fn new() -> Self {
        Self {
            n: AtomicUsize::new(0),
            abort: AtomicBool::new(false),
            detail: Mutex::new(Vec::new()),
        }
    }

    /// Record one unit's failure; returns the new total. Detail beyond
    /// [`MAX_RETAINED`] is dropped, the count keeps rising.
    pub fn record(&self, unit: &str, error: String) -> usize {
        let n = self.n.fetch_add(1, Ordering::Relaxed) + 1;
        let mut d = self.detail.lock().unwrap_or_else(|e| e.into_inner());
        if d.len() < MAX_RETAINED {
            d.push(UnitFailure {
                unit: unit.to_string(),
                error,
            });
        }
        n
    }

    /// Stop the pass: the failure cap tripped, so remaining units are
    /// abandoned rather than burning GPU on a systemic fault.
    pub fn set_abort(&self) {
        self.abort.store(true, Ordering::Relaxed);
    }

    pub fn aborted(&self) -> bool {
        self.abort.load(Ordering::Relaxed)
    }

    pub fn into_report(self, n_units: usize) -> IngestReport {
        IngestReport {
            n_units,
            n_failed: self.n.load(Ordering::Relaxed),
            aborted: self.abort.load(Ordering::Relaxed),
            failures: self.detail.into_inner().unwrap_or_else(|e| e.into_inner()),
        }
    }
}

fn slot() -> &'static RwLock<BTreeMap<String, IngestReport>> {
    static SLOT: OnceLock<RwLock<BTreeMap<String, IngestReport>>> = OnceLock::new();
    SLOT.get_or_init(|| RwLock::new(BTreeMap::new()))
}

/// Publish a pass's outcome under its name (`repo_map`, `code_read`, …). A
/// later run of the SAME pass replaces its entry — the question the API answers
/// is "is each map complete *now*" — while other passes keep their own state.
pub fn publish(pass: &str, report: IngestReport) {
    if let Ok(mut g) = slot().write() {
        g.insert(pass.to_string(), report);
    }
}

/// Every pass that has run, keyed by name. Empty before any pass completes.
///
/// Callers that derive aggregates (the API's `incomplete` flag) do so over this
/// one snapshot rather than through per-question helpers here — a helper would
/// take a second snapshot that can disagree with the pass list it is answering
/// about.
pub fn latest() -> BTreeMap<String, IngestReport> {
    slot().read().map(|g| g.clone()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clean_pass_is_complete() {
        let f = Failures::new();
        let r = f.into_report(10);
        assert_eq!(r.n_failed, 0);
        assert!(!r.aborted);
        assert!(!r.is_incomplete(), "no failures => complete");
    }

    /// Failures under the cap still mark the map incomplete — the exact case
    /// that used to pass silently as "ingest complete".
    #[test]
    fn failures_under_the_cap_still_mark_incomplete() {
        let f = Failures::new();
        assert_eq!(f.record("a/", "boom".into()), 1);
        assert_eq!(f.record("b/", "boom".into()), 2);
        let r = f.into_report(10);
        assert_eq!(r.n_failed, 2);
        assert!(!r.aborted);
        assert!(r.is_incomplete(), "a partial map must report incomplete");
        assert_eq!(r.failures.len(), 2);
        assert_eq!(r.failures[0].unit, "a/");
        assert_eq!(r.failures[0].error, "boom");
    }

    /// The count stays exact past the retention cap; only detail truncates.
    #[test]
    fn detail_truncates_but_the_count_is_exact() {
        let f = Failures::new();
        for i in 0..(MAX_RETAINED + 17) {
            f.record(&format!("d{i}/"), "e".into());
        }
        let r = f.into_report(400);
        assert_eq!(r.n_failed, MAX_RETAINED + 17);
        assert_eq!(r.failures.len(), MAX_RETAINED);
    }

    #[test]
    fn abort_is_reported_and_stops_workers() {
        let f = Failures::new();
        assert!(!f.aborted());
        f.set_abort();
        assert!(f.aborted());
        let r = f.into_report(300);
        assert!(r.aborted);
        assert!(r.is_incomplete());
    }
}

#[cfg(test)]
mod publish_tests {
    use super::*;

    /// Passes are independent: publishing `code_read` must not erase what
    /// `repo_map` reported, or the API would answer for whichever ran last.
    #[test]
    fn passes_are_keyed_independently() {
        let mut clean = Failures::new().into_report(5);
        clean.n_units = 5;
        publish("pass_a_test", clean);

        let bad = Failures::new();
        bad.record("x/", "kaboom".into());
        publish("pass_b_test", bad.into_report(9));

        let all = latest();
        assert!(!all["pass_a_test"].is_incomplete());
        assert!(all["pass_b_test"].is_incomplete());
        // The API's aggregate, as it computes it — any-of over one snapshot.
        assert!(
            all.values().any(|r| r.is_incomplete()),
            "one bad pass makes the whole map suspect"
        );
    }
}
