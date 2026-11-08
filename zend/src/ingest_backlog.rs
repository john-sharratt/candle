//! Cross-layer, cross-pass backlog of pending background-ingest work — the
//! GUI's single merged "N remaining, last: `<path>`" indicator (`GET
//! /v1/status`).
//!
//! **Not [`crate::loading::LoadProgress`].** That struct is the daemon's
//! one-way startup state machine (`mark_ready` is permanent) and reports ONE
//! linear step's `(progressed, total)` at a time; this counter empties and
//! refills indefinitely while the daemon stays `ready`, spans both ingest
//! layers, uploads and the calibration sweep at once
//! (`crate::calibrate::calibrate_layer`, its units named `<path> (calibrated)`),
//! and counts only units that are actually RUN — a unit already committed is
//! not backlog.
//!
//! **Reading and calibrating are counted apart** ([`Work`]) and reported
//! together: the bar's fill is all of it, and the calibration units still to
//! go are named on their own, so a sweep of 24-question calibrations — which
//! costs many times the reading it follows — reads as calibration and not as
//! indexing. Each kind settles to empty on its own, so one kind's batch ending
//! never clears the other's count.
//!
//! A process-global, mirroring [`crate::ingest_report`]'s `publish`/`latest`
//! shape: the counter is written from deep inside `repo_scan::run_dir_pool`
//! and `code_read::ingest_jobs` and read by the status API, with nothing in
//! between that could usefully hold an `Arc` to it instead.

use std::sync::{Mutex, OnceLock};

/// What a unit of backlog is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Work {
    /// A file, a folder or a derived unit read into its layer.
    Reading,
    /// A read unit asked for its questions and calibrated with them.
    Calibrating,
}

/// Snapshot for the API. `total == 0` never reaches the wire — see
/// [`IngestBacklog::snapshot`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BacklogSnapshot {
    /// Every kind of work, together.
    pub processed: u64,
    pub total: u64,
    pub last_item: Option<String>,
    /// Calibration units still to go — part of `total - processed`.
    pub calibrating: u64,
}

#[derive(Default)]
struct Count {
    processed: u64,
    total: u64,
}

impl Count {
    /// Reached to zero the moment `processed` catches up with `total`, so the
    /// next batch of work always starts from a clean `0/0` rather than a
    /// total that only ever grows across unrelated batches.
    fn settle(&mut self) {
        if self.processed >= self.total {
            *self = Count::default();
        }
    }

    fn left(&self) -> u64 {
        self.total.saturating_sub(self.processed)
    }
}

#[derive(Default)]
struct BacklogState {
    reading: Count,
    calibrating: Count,
    last_item: Option<String>,
}

impl BacklogState {
    fn count(&mut self, work: Work) -> &mut Count {
        match work {
            Work::Reading => &mut self.reading,
            Work::Calibrating => &mut self.calibrating,
        }
    }

    /// Settle `work`'s count, and forget the last item once nothing of either
    /// kind is pending.
    fn settle(&mut self, work: Work) {
        self.count(work).settle();
        if self.reading.total == 0 && self.calibrating.total == 0 {
            self.last_item = None;
        }
    }
}

pub struct IngestBacklog {
    inner: Mutex<BacklogState>,
}

impl Default for IngestBacklog {
    fn default() -> Self {
        Self::new()
    }
}

impl IngestBacklog {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(BacklogState::default()),
        }
    }

    /// Register `n` units of `work` this pass will really run (resume-cache
    /// hits excluded by the caller before this is reached).
    pub fn add_pending(&self, work: Work, n: u64) {
        if n == 0 {
            return;
        }
        let mut s = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        s.count(work).total += n;
        s.settle(work);
    }

    /// One registered unit of `work` finished — succeeded, tolerated-failed,
    /// or was cut short by a shutdown cancel. Every registered unit must call
    /// this exactly once on every exit path, or the backlog wedges non-empty.
    pub fn item_done(&self, work: Work, item: &str) {
        let mut s = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        s.count(work).processed += 1;
        s.last_item = Some(item.to_string());
        s.settle(work);
    }

    /// `n` registered units of `work` will never run (a pass aborted, or a
    /// shutdown cancel left them claimed but unstarted) — hand them back so
    /// the backlog can still reach zero.
    pub fn drop_pending(&self, work: Work, n: u64) {
        if n == 0 {
            return;
        }
        let mut s = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let count = s.count(work);
        count.total = count.total.saturating_sub(n);
        s.settle(work);
    }

    /// `None` when nothing is pending — the GUI hides the bar on this.
    pub fn snapshot(&self) -> Option<BacklogSnapshot> {
        let s = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let total = s.reading.total + s.calibrating.total;
        if total == 0 {
            return None;
        }
        Some(BacklogSnapshot {
            processed: s.reading.processed + s.calibrating.processed,
            total,
            last_item: s.last_item.clone(),
            calibrating: s.calibrating.left(),
        })
    }
}

/// The one live backlog. Deep pool call sites reach it without threading a
/// parameter through already-large signatures, exactly as
/// `ingest_report::publish` is reached.
pub fn global() -> &'static IngestBacklog {
    static SLOT: OnceLock<IngestBacklog> = OnceLock::new();
    SLOT.get_or_init(IngestBacklog::new)
}

pub fn add_pending(work: Work, n: u64) {
    global().add_pending(work, n);
}

pub fn item_done(work: Work, item: &str) {
    global().item_done(work, item);
}

pub fn drop_pending(work: Work, n: u64) {
    global().drop_pending(work, n);
}

pub fn snapshot() -> Option<BacklogSnapshot> {
    global().snapshot()
}

#[cfg(test)]
mod tests {
    use super::Work::{Calibrating, Reading};
    use super::*;

    #[test]
    fn fills_then_resets_to_empty_on_catch_up() {
        let b = IngestBacklog::new();
        assert_eq!(b.snapshot(), None);
        b.add_pending(Reading, 3);
        assert_eq!(
            b.snapshot(),
            Some(BacklogSnapshot {
                processed: 0,
                total: 3,
                last_item: None,
                calibrating: 0,
            })
        );
        b.item_done(Reading, "a.rs");
        b.item_done(Reading, "b.rs");
        assert_eq!(
            b.snapshot(),
            Some(BacklogSnapshot {
                processed: 2,
                total: 3,
                last_item: Some("b.rs".to_string()),
                calibrating: 0,
            })
        );
        b.item_done(Reading, "c.rs");
        assert_eq!(b.snapshot(), None, "processed caught total => empty");
    }

    #[test]
    fn a_second_pass_starts_from_zero() {
        let b = IngestBacklog::new();
        b.add_pending(Reading, 1);
        b.item_done(Reading, "a.rs");
        assert_eq!(b.snapshot(), None);
        b.add_pending(Reading, 2);
        assert_eq!(
            b.snapshot(),
            Some(BacklogSnapshot {
                processed: 0,
                total: 2,
                last_item: None,
                calibrating: 0,
            }),
            "a later pass must not inherit the prior pass's counts",
        );
    }

    #[test]
    fn overlapping_passes_merge_into_one_total() {
        // Two layers (or a layer + an upload) registering work before either
        // finishes must merge into one number, not clobber each other.
        let b = IngestBacklog::new();
        b.add_pending(Reading, 2); // repo_map
        b.add_pending(Reading, 3); // code_reading
        assert_eq!(b.snapshot().unwrap().total, 5);
        for f in ["a", "b", "c", "d"] {
            b.item_done(Reading, f);
        }
        assert_eq!(b.snapshot().unwrap().processed, 4);
        b.item_done(Reading, "e");
        assert_eq!(b.snapshot(), None);
    }

    #[test]
    fn total_can_grow_mid_drain_and_the_fill_can_slip_back() {
        let b = IngestBacklog::new();
        b.add_pending(Reading, 2);
        b.item_done(Reading, "a");
        assert_eq!(
            b.snapshot().unwrap(),
            BacklogSnapshot {
                processed: 1,
                total: 2,
                last_item: Some("a".to_string()),
                calibrating: 0,
            }
        );
        // A file changed mid-drain: more work, not a miscount.
        b.add_pending(Reading, 3);
        assert_eq!(b.snapshot().unwrap().total, 5);
        assert_eq!(b.snapshot().unwrap().processed, 1);
    }

    #[test]
    fn drop_pending_abandons_and_can_settle_to_empty() {
        let b = IngestBacklog::new();
        b.add_pending(Reading, 3);
        b.item_done(Reading, "a");
        // The remaining 2 were claimed but never run (abort / shutdown).
        b.drop_pending(Reading, 2);
        assert_eq!(b.snapshot(), None, "abandoned work must still reach zero");
    }

    #[test]
    fn drop_pending_saturates_rather_than_underflowing() {
        let b = IngestBacklog::new();
        b.drop_pending(Reading, 5); // nothing pending yet — must not panic or go negative
        assert_eq!(b.snapshot(), None);
    }

    #[test]
    fn snapshot_is_none_when_empty() {
        let b = IngestBacklog::new();
        assert_eq!(b.snapshot(), None);
        b.add_pending(Calibrating, 0); // a zero registration is a no-op
        assert_eq!(b.snapshot(), None);
    }

    /// **A calibration sweep is backlog in its own name**: the fill is the
    /// reading and the calibrating together, and the calibration units still
    /// to go are reported on their own.
    #[test]
    fn calibration_is_counted_with_the_reading_and_named_apart() {
        let b = IngestBacklog::new();
        b.add_pending(Reading, 2);
        b.add_pending(Calibrating, 5);
        b.item_done(Calibrating, "a.rs (calibrated)");
        assert_eq!(
            b.snapshot(),
            Some(BacklogSnapshot {
                processed: 1,
                total: 7,
                last_item: Some("a.rs (calibrated)".to_string()),
                calibrating: 4,
            })
        );
        b.item_done(Reading, "b.rs");
        b.item_done(Reading, "c.rs");
        assert_eq!(
            b.snapshot(),
            Some(BacklogSnapshot {
                processed: 1,
                total: 5,
                last_item: Some("c.rs".to_string()),
                calibrating: 4,
            }),
            "the reading settled on its own; the sweep's count stands",
        );
    }

    /// **One kind's batch ending never clears the other's**: a reading pass
    /// that catches up while a sweep runs leaves the sweep's count whole —
    /// a shared counter emptied the sweep's bar the moment the two met.
    #[test]
    fn a_reading_pass_settling_leaves_the_sweep_counted() {
        let b = IngestBacklog::new();
        b.add_pending(Calibrating, 3);
        b.add_pending(Reading, 1);
        b.item_done(Reading, "a.rs");
        b.item_done(Reading, "stray.rs");
        assert_eq!(b.snapshot().unwrap().calibrating, 3);
        for _ in 0..3 {
            b.item_done(Calibrating, "x (calibrated)");
        }
        assert_eq!(b.snapshot(), None);
    }
}
