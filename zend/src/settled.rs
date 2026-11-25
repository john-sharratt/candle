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
//! **Empty is not done until something has counted.** At boot every count is
//! empty because nothing has run: the daemon is loading, and the first ingest
//! pass — which registers the backlog and whose warm-up learns every file's
//! backgrounds — has not begun. So the verdict also waits for the load to end
//! and for that first pass and its warm-up to finish
//! ([`crate::ingest_worker::first_pass_warmed`]).
//!
//! The verdict is `settled` only when all of it is done, and names what it
//! waits on otherwise. It reads counters and flags and nothing else, so
//! `/v1/status` answers it at once.

use serde::Serialize;

use crate::ingest_backlog::BacklogSnapshot;

/// The verdict, and what it waits on when it is not settled.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Settled {
    pub settled: bool,
    pub waiting_on: Vec<String>,
}

/// What the verdict is made of.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Drains<'a> {
    /// The daemon is still loading.
    pub loading: bool,
    /// The process's first ingest pass and its warm-up have finished.
    pub first_pass_warmed: bool,
    pub backlog: Option<&'a BacklogSnapshot>,
    pub calibration_owed: usize,
    pub warming: Option<(usize, usize)>,
    pub normalizing: Option<(usize, usize)>,
}

/// The verdict from what drains.
pub fn settled(d: &Drains<'_>) -> Settled {
    let mut waiting_on = Vec::new();
    if d.loading {
        waiting_on.push("loading".to_string());
    }
    if !d.first_pass_warmed {
        waiting_on.push("the first ingest pass and its warm-up".to_string());
    }
    let (backlog, calibration_owed, warming, normalizing) =
        (d.backlog, d.calibration_owed, d.warming, d.normalizing);
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

    /// A member past its load and its first pass, with nothing draining.
    fn done<'a>() -> Drains<'a> {
        Drains {
            first_pass_warmed: true,
            ..Drains::default()
        }
    }

    /// **Settled only when every drain is empty**, and every one still
    /// draining is named.
    #[test]
    fn settled_only_when_everything_has_drained() {
        assert_eq!(
            settled(&done()),
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
            settled(&Drains {
                backlog: Some(&b),
                calibration_owed: 2,
                warming: Some((5, 16)),
                normalizing: Some((1, 64)),
                ..done()
            }),
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
            !settled(&Drains {
                calibration_owed: 1,
                ..done()
            })
            .settled,
            "a failed calibration alone"
        );
    }

    /// **Empty counts at boot are not settled**: a member still loading, or
    /// whose first pass and warm-up have not finished, has counted nothing
    /// yet.
    #[test]
    fn nothing_counted_yet_is_not_settled() {
        assert_eq!(
            settled(&Drains::default()).waiting_on,
            vec!["the first ingest pass and its warm-up".to_string()]
        );
        assert_eq!(
            settled(&Drains {
                loading: true,
                ..Drains::default()
            })
            .waiting_on,
            vec![
                "loading".to_string(),
                "the first ingest pass and its warm-up".to_string()
            ]
        );
    }
}
