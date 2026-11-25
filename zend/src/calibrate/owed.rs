//! What each layer's calibration still owes: the units its last pass found
//! without a committed calibration, less those committed since.
//!
//! The backlog's calibration count (`crate::ingest_backlog`) is the sweep
//! running — a unit leaves it once tried, whether its calibration sealed or
//! failed. A failed unit is calibrated again only by a later pass, so between
//! passes it is calibrated nowhere and counted nowhere. This ledger holds it:
//! a pass sets its layer's count from its plan, every unit sealed here or by
//! another member pays one off, and a failure pays nothing. `/v1/status`
//! reads the sum, and the memory is not settled while it is above zero.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Units owed a calibration, by layer.
#[derive(Default)]
pub struct Owed {
    by_layer: Mutex<HashMap<String, usize>>,
}

impl Owed {
    /// `layer`'s pass planned `n` units without a committed calibration: the
    /// ready ones it runs and the ones it waits on.
    pub fn planned(&self, layer: &str, n: usize) {
        let mut owed = self.by_layer.lock().unwrap_or_else(|e| e.into_inner());
        owed.insert(layer.to_string(), n);
    }

    /// One of `layer`'s units committed a calibration.
    pub fn committed(&self, layer: &str) {
        let mut owed = self.by_layer.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(n) = owed.get_mut(layer) {
            *n = n.saturating_sub(1);
        }
    }

    /// Units owed across every layer.
    pub fn total(&self) -> usize {
        let owed = self.by_layer.lock().unwrap_or_else(|e| e.into_inner());
        owed.values().sum()
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
        o.planned("code_reading", 3);
        o.planned("repo_map", 1);
        o.committed("code_reading");
        o.committed("code_reading");
        assert_eq!(o.total(), 2, "one code_reading unit failed, repo_map waits");
        o.planned("code_reading", 1);
        o.committed("code_reading");
        o.committed("repo_map");
        assert_eq!(o.total(), 0);
        o.committed("repo_map");
        assert_eq!(o.total(), 0, "saturates, never underflows");
    }
}
