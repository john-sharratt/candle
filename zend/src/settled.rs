//! Whether the memory is settled: everything read, faced and calibrated, and
//! every score background learned — the state a measurement of retrieval needs
//! the member in before it asks anything (`zend/examples/dependency_battery`).
//!
//! Four things drain after a deploy or a moved branch, and each is counted
//! where it runs:
//!
//! - the ingest backlog — files read, faced and calibrated by the sweep
//!   running ([`crate::ingest_backlog`]);
//! - calibrations owed — units a calibration pass set out to calibrate, a
//!   failed one included, that no pass has committed since
//!   ([`crate::calibrate::owed`]); a unit no pass can calibrate yet is
//!   reported apart and not waited on;
//! - the ingest normalization warm-up — the per-file backgrounds of files a
//!   pass read ([`candle_conversation::IngestWarmer`]);
//! - the dialogue normalization replay after a load.
//!
//! The verdict is `settled` only when all four are empty, and names what it
//! waits on otherwise. It reads counters and nothing else, so `/v1/status`
//! answers it at once.

use serde::Serialize;

use crate::ingest_backlog::BacklogSnapshot;

/// The verdict, and what it waits on when it is not settled.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Settled {
    pub settled: bool,
    pub waiting_on: Vec<String>,
}

/// The verdict from the four counts.
pub fn settled(
    backlog: Option<&BacklogSnapshot>,
    calibration_owed: usize,
    warming: Option<(usize, usize)>,
    normalizing: Option<(usize, usize)>,
) -> Settled {
    let mut waiting_on = Vec::new();
    if let Some(b) = backlog {
        waiting_on.push(format!(
            "ingest: {} of {} done ({} calibrating, {} facing)",
            b.processed, b.total, b.calibrating, b.facing
        ));
    }
    if calibration_owed > 0 {
        waiting_on.push(format!("calibration: {calibration_owed} owed"));
    }
    if let Some((done, total)) = warming {
        waiting_on.push(format!("warming: {done} of {total} files"));
    }
    if let Some((done, total)) = normalizing {
        waiting_on.push(format!("normalizing: {done} of {total} turns"));
    }
    Settled {
        settled: waiting_on.is_empty(),
        waiting_on,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Settled only when all four are empty**, and every one still draining
    /// is named.
    #[test]
    fn settled_only_when_all_four_are_empty() {
        assert_eq!(
            settled(None, 0, None, None),
            Settled {
                settled: true,
                waiting_on: vec![],
            }
        );
        let b = BacklogSnapshot {
            processed: 3,
            total: 10,
            last_item: None,
            calibrating: 4,
            facing: 1,
        };
        assert_eq!(
            settled(Some(&b), 2, Some((5, 16)), Some((1, 64))),
            Settled {
                settled: false,
                waiting_on: vec![
                    "ingest: 3 of 10 done (4 calibrating, 1 facing)".to_string(),
                    "calibration: 2 owed".to_string(),
                    "warming: 5 of 16 files".to_string(),
                    "normalizing: 1 of 64 turns".to_string(),
                ],
            }
        );
        assert!(
            !settled(None, 1, None, None).settled,
            "a failed calibration alone"
        );
    }
}
