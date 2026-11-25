//! What each layer's calibration still owes: the units its last pass set out
//! to calibrate, less those committed since — and, apart, the units it found
//! it could not calibrate yet.
//!
//! The backlog's calibration count (`crate::ingest_backlog`) is the sweep
//! running — a unit leaves it once tried, whether its calibration sealed or
//! failed. A failed unit is calibrated again only by a later pass, so between
//! passes it is calibrated nowhere and counted nowhere. This ledger holds it:
//! a pass sets its layer's count from the units it runs, every unit sealed
//! here or by another member pays one off, and a failure pays nothing.
//! `/v1/status` reads the sum, and the memory is not settled while it is
//! above zero.
//!
//! **A unit the pass waits on is not owed.** It is not ready — its reading is
//! of an older version, its chain broken, a turn not yet held, or its failures
//! past the limit — and no pass calibrates it until that changes, which may be
//! never. Counted as owed, one such unit would keep the memory unsettled for
//! good. It is counted apart, as `waiting`, and reported beside the verdict;
//! a unit that becomes ready is run, and owed, by the pass that finds it so.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// A layer's count: units owed, and units waited on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Count {
    owed: usize,
    waiting: usize,
}

/// Units owed a calibration, and units waited on, by layer.
#[derive(Default)]
pub struct Owed {
    by_layer: Mutex<HashMap<String, Count>>,
}

impl Owed {
    /// `layer`'s pass runs `run` units without a committed calibration and
    /// waits on `waiting` it cannot calibrate yet.
    pub fn planned(&self, layer: &str, run: usize, waiting: usize) {
        let mut by_layer = self.by_layer.lock().unwrap_or_else(|e| e.into_inner());
        by_layer.insert(layer.to_string(), Count { owed: run, waiting });
    }

    /// One of `layer`'s units committed a calibration.
    pub fn committed(&self, layer: &str) {
        let mut by_layer = self.by_layer.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(c) = by_layer.get_mut(layer) {
            c.owed = c.owed.saturating_sub(1);
        }
    }

    /// Units owed across every layer.
    pub fn total(&self) -> usize {
        let by_layer = self.by_layer.lock().unwrap_or_else(|e| e.into_inner());
        by_layer.values().map(|c| c.owed).sum()
    }

    /// Units waited on across every layer.
    pub fn waiting(&self) -> usize {
        let by_layer = self.by_layer.lock().unwrap_or_else(|e| e.into_inner());
        by_layer.values().map(|c| c.waiting).sum()
    }
}

/// The one live ledger, reached from the calibration pass and the status API
/// as [`crate::ingest_backlog`] is.
pub fn global() -> &'static Owed {
    static SLOT: OnceLock<Owed> = OnceLock::new();
    SLOT.get_or_init(Owed::default)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A failed unit stays owed until a pass commits it**: the plan sets
    /// the count, each commit pays one, a failure pays none, and the next
    /// pass's plan replaces the layer's count.
    #[test]
    fn a_failure_stays_owed_until_a_pass_commits_it() {
        let o = Owed::default();
        assert_eq!(o.total(), 0);
        o.planned("code_reading", 3, 0);
        o.planned("repo_map", 1, 0);
        o.committed("code_reading");
        o.committed("code_reading");
        assert_eq!(o.total(), 2, "one code_reading unit failed, repo_map runs");
        o.planned("code_reading", 1, 0);
        o.committed("code_reading");
        o.committed("repo_map");
        assert_eq!(o.total(), 0);
        o.committed("repo_map");
        assert_eq!(o.total(), 0, "saturates, never underflows");
    }

    /// **A unit the pass waits on is counted apart and owes nothing**: a
    /// layer whose every runnable unit committed is settled whatever it waits
    /// on.
    #[test]
    fn a_waited_unit_is_counted_apart() {
        let o = Owed::default();
        o.planned("code_reading", 1, 22);
        assert_eq!((o.total(), o.waiting()), (1, 22));
        o.committed("code_reading");
        assert_eq!((o.total(), o.waiting()), (0, 22));
        o.planned("code_reading", 0, 21);
        assert_eq!(o.waiting(), 21, "the next plan replaces the count");
    }
}
