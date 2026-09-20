//! Cross-layer, cross-pass backlog of pending background-ingest work — the
//! GUI's single merged "N remaining, last: `<path>`" indicator (`GET
//! /v1/status`).
//!
//! **Not [`crate::loading::LoadProgress`].** That struct is the daemon's
//! one-way startup state machine (`mark_ready` is permanent) and reports ONE
//! linear step's `(progressed, total)` at a time; this counter empties and
//! refills indefinitely while the daemon stays `ready`, spans both ingest
//! layers plus uploads at once, and counts only units that are actually RUN —
//! a resume-cache hit is not backlog.
//!
//! A process-global, mirroring [`crate::ingest_report`]'s `publish`/`latest`
//! shape: the counter is written from deep inside `repo_scan::run_dir_pool`
//! and `code_read::run_file_pool` and read by the status API, with nothing in
//! between that could usefully hold an `Arc` to it instead.

use std::sync::{Mutex, OnceLock};

/// Snapshot for the API. `total == 0` never reaches the wire — see
/// [`IngestBacklog::snapshot`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BacklogSnapshot {
    pub processed: u64,
    pub total: u64,
    pub last_item: Option<String>,
}

#[derive(Default)]
struct BacklogState {
    processed: u64,
    total: u64,
    last_item: Option<String>,
}

/// Reached to zero the moment `processed` catches up with `total`, so the
/// next batch of work always starts from a clean `0/0` rather than carrying
/// a stale `last_item` or a total that only ever grows across unrelated
/// batches.
fn settle(s: &mut BacklogState) {
    if s.processed >= s.total {
        s.processed = 0;
        s.total = 0;
        s.last_item = None;
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

    /// Register `n` units this pass will really ingest (resume-cache hits
    /// excluded by the caller before this is reached).
    pub fn add_pending(&self, n: u64) {
        if n == 0 {
            return;
        }
        let mut s = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        s.total += n;
        settle(&mut s);
    }

    /// One registered unit finished — succeeded, tolerated-failed, or was
    /// cut short by a shutdown cancel. Every registered unit must call this
    /// exactly once on every exit path, or the backlog wedges non-empty.
    pub fn item_done(&self, item: &str) {
        let mut s = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        s.processed += 1;
        s.last_item = Some(item.to_string());
        settle(&mut s);
    }

    /// `n` registered units will never run (a pass aborted, or a shutdown
    /// cancel left them claimed but unstarted) — hand them back so the
    /// backlog can still reach zero.
    pub fn drop_pending(&self, n: u64) {
        if n == 0 {
            return;
        }
        let mut s = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        s.total = s.total.saturating_sub(n);
        settle(&mut s);
    }

    /// `None` when nothing is pending — the GUI hides the bar on this.
    pub fn snapshot(&self) -> Option<BacklogSnapshot> {
        let s = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if s.total == 0 {
            return None;
        }
        Some(BacklogSnapshot {
            processed: s.processed,
            total: s.total,
            last_item: s.last_item.clone(),
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

pub fn add_pending(n: u64) {
    global().add_pending(n);
}

pub fn item_done(item: &str) {
    global().item_done(item);
}

pub fn drop_pending(n: u64) {
    global().drop_pending(n);
}

pub fn snapshot() -> Option<BacklogSnapshot> {
    global().snapshot()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_then_resets_to_empty_on_catch_up() {
        let b = IngestBacklog::new();
        assert_eq!(b.snapshot(), None);
        b.add_pending(3);
        assert_eq!(
            b.snapshot(),
            Some(BacklogSnapshot {
                processed: 0,
                total: 3,
                last_item: None
            })
        );
        b.item_done("a.rs");
        b.item_done("b.rs");
        assert_eq!(
            b.snapshot(),
            Some(BacklogSnapshot {
                processed: 2,
                total: 3,
                last_item: Some("b.rs".to_string())
            })
        );
        b.item_done("c.rs");
        assert_eq!(b.snapshot(), None, "processed caught total => empty");
    }

    #[test]
    fn a_second_pass_starts_from_zero() {
        let b = IngestBacklog::new();
        b.add_pending(1);
        b.item_done("a.rs");
        assert_eq!(b.snapshot(), None);
        b.add_pending(2);
        assert_eq!(
            b.snapshot(),
            Some(BacklogSnapshot {
                processed: 0,
                total: 2,
                last_item: None
            }),
            "a later pass must not inherit the prior pass's counts",
        );
    }

    #[test]
    fn overlapping_passes_merge_into_one_total() {
        // Two layers (or a layer + an upload) registering work before either
        // finishes must merge into one number, not clobber each other.
        let b = IngestBacklog::new();
        b.add_pending(2); // repo_map
        b.add_pending(3); // code_reading
        assert_eq!(b.snapshot().unwrap().total, 5);
        for f in ["a", "b", "c", "d"] {
            b.item_done(f);
        }
        assert_eq!(b.snapshot().unwrap().processed, 4);
        b.item_done("e");
        assert_eq!(b.snapshot(), None);
    }

    #[test]
    fn total_can_grow_mid_drain_and_the_fill_can_slip_back() {
        let b = IngestBacklog::new();
        b.add_pending(2);
        b.item_done("a");
        assert_eq!(
            b.snapshot().unwrap(),
            BacklogSnapshot {
                processed: 1,
                total: 2,
                last_item: Some("a".to_string()),
            }
        );
        // A file changed mid-drain: more work, not a miscount.
        b.add_pending(3);
        assert_eq!(b.snapshot().unwrap().total, 5);
        assert_eq!(b.snapshot().unwrap().processed, 1);
    }

    #[test]
    fn drop_pending_abandons_and_can_settle_to_empty() {
        let b = IngestBacklog::new();
        b.add_pending(3);
        b.item_done("a");
        // The remaining 2 were claimed but never run (abort / shutdown).
        b.drop_pending(2);
        assert_eq!(b.snapshot(), None, "abandoned work must still reach zero");
    }

    #[test]
    fn drop_pending_saturates_rather_than_underflowing() {
        let b = IngestBacklog::new();
        b.drop_pending(5); // nothing pending yet — must not panic or go negative
        assert_eq!(b.snapshot(), None);
    }

    #[test]
    fn snapshot_is_none_when_empty() {
        let b = IngestBacklog::new();
        assert_eq!(b.snapshot(), None);
        b.add_pending(0); // a zero registration is a no-op
        assert_eq!(b.snapshot(), None);
    }
}
