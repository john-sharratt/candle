//! A task cut off by a restart is started again.
//!
//! A worker's record says it runs for as long as its member beats on it
//! ([`super::threads::BEAT_LOST_MS`]). When the member restarts, nothing ends the
//! task: its conversation keeps every turn it sealed, and until now the work
//! stopped there for someone to notice. A deploy lost a job that had read
//! everything it needed and written nothing yet (2026-10-31). So every member
//! watches for lost tasks and one of them, by lease, continues each in its own
//! conversation.

use std::sync::{Arc, Weak};
use std::time::Duration;

use tokio::runtime::Handle;

use super::archiving::{lost, LOST_ARCHIVE_MS};
use super::threads::{self, Worker};
use super::workers;
use crate::ingest_lease::{self, Share};
use crate::session::ZendSession;

/// How often lost tasks are looked for.
const EVERY: Duration = Duration::from_secs(60);

/// Restarts a task is carried through. Past this something other than a
/// restart is ending it, and it stays lost for a person to read.
pub const MAX_RESUMES: u32 = 3;

/// How long a lost task's beat has been silent before it is started again.
/// "Lost" is read off a record that replicates, so a member whose copy lags, or
/// whose runtime stalls, sees a live task's beat stop for a while; started again
/// then, the task would run twice, each copy writing its own turns. Well past
/// [`super::threads::BEAT_LOST_MS`], and a restart's load is within it.
pub const RESUME_AFTER_MS: u64 = 10 * 60 * 1000;

/// Whether `w`, as observed at `now`, should be started again. A part is not:
/// the worker that started it is, and starts what it still needs. One silent
/// for as long as the archive sweep waits has been given up on.
pub fn due(w: &Worker, now: u64) -> bool {
    let silent = now.saturating_sub(w.beat_ms);
    lost(w)
        && !w.is_part()
        && w.resumes < MAX_RESUMES
        && (RESUME_AFTER_MS..LOST_ARCHIVE_MS).contains(&silent)
}

/// Start again, on one member, every lost task. How many this one took.
fn pass(session: &Arc<ZendSession>, now: u64) -> usize {
    let Some(records) = session.shared_substrate() else {
        return 0;
    };
    let mut resumed = 0;
    for (owner, t) in threads::every_thread(&records) {
        for w in threads::workers(&records, &owner, &t.name) {
            let w = w.observed(now);
            if !due(&w, now) {
                continue;
            }
            // Held across the resume: a member that loses this finds the task
            // running on the next pass, or takes it if this one failed.
            let lease = format!("resume/{}/{}", w.conv_id, w.resumes);
            let Share::Mine(_guard) = ingest_lease::share(&lease) else {
                continue;
            };
            match workers::resume(session, &owner, &t.name, &w.name, now) {
                Ok(true) => resumed += 1,
                Ok(false) => {}
                Err(e) => tracing::warn!(
                    conv_id = %w.conv_id,
                    "mcp: a task cut off by a restart could not be started again: {e}"
                ),
            }
        }
    }
    resumed
}

/// Look for lost tasks every [`EVERY`] for as long as the session lives.
async fn resuming(session: Weak<ZendSession>) {
    let mut every = tokio::time::interval(EVERY);
    loop {
        every.tick().await;
        let Some(session) = session.upgrade() else {
            return;
        };
        if !session.is_loaded() {
            continue;
        }
        let at = workers::now_ms();
        let handle = session.clone();
        match tokio::task::spawn_blocking(move || pass(&handle, at)).await {
            Ok(0) => {}
            Ok(tasks) => tracing::info!(tasks, "mcp: started again what a restart cut off"),
            Err(e) => tracing::warn!("mcp: looking for lost tasks failed: {e}"),
        }
    }
}

/// Start [`resuming`] on `runtime` for `session`.
pub fn start_resuming(session: &Arc<ZendSession>, runtime: &Handle) {
    runtime.spawn(resuming(Arc::downgrade(session)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::threads::{State, BEAT_LOST_MS};

    fn worker(name: &str, resumes: u32) -> Worker {
        let mut w: Worker = serde_json::from_value(serde_json::json!({
            "thread": "t", "name": name, "conv_id": "c", "prompt": "p",
            "state": "running", "role": "admin", "priority": "normal",
            "started_ms": 0, "turn_ms": 0, "beat_ms": 1_000,
            "activity": null, "no_entry": null,
        }))
        .unwrap();
        w.resumes = resumes;
        w
    }

    #[test]
    fn only_a_task_lost_lately_and_not_too_often_is_resumed() {
        let after = 1_000 + RESUME_AFTER_MS;
        assert!(due(&worker("a", 0).observed(after), after));
        assert!(
            !due(&worker("a", 0).observed(1_000), 1_000),
            "its beat is fresh"
        );
        let lagging = 1_001 + BEAT_LOST_MS;
        assert!(lost(&worker("a", 0).observed(lagging)), "it reads as lost");
        assert!(
            !due(&worker("a", 0).observed(lagging), lagging),
            "but a member's copy may only be behind"
        );
        assert!(
            !due(&worker("a", MAX_RESUMES).observed(after), after),
            "carried through enough restarts"
        );
        assert!(
            !due(&worker("a/part", 0).observed(after), after),
            "its worker starts its parts"
        );
        let late = 1_000 + LOST_ARCHIVE_MS;
        assert!(!due(&worker("a", 0).observed(late), late), "given up on");
        assert_eq!(worker("a", 0).observed(1_000).state, State::Running);
    }
}
