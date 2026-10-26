//! Archiving threads and their workers (`docs/mcp_guides.md` §3): by
//! `thread_archive`, and by the sweep once they have sat idle a day.
//!
//! A worker that finished its task is kept whole — its conversation and its
//! KV — so Claude can continue it with `task_reply`, as it can a worker that
//! asked or is stuck. Archiving distils the conversation to text
//! (`ZendSession::archive_worker`), which is one way: an archived worker's
//! evidence stays readable, and the worker is never continued.
//!
//! The sweep archives a worker nothing has touched for [`IDLE_MS`], and a
//! thread whose newest activity is that old with nothing running on it. It
//! touches only these workers: a person's own zend conversations are never
//! archived but by that person.

use std::sync::{Arc, Weak};
use std::time::Duration;

use tokio::runtime::Handle;

use super::records::Records;
use super::threads::{self, Note, State, Thread, Worker};
use super::workers::now_ms;
use crate::session::ZendSession;

/// How long a worker or a thread sits idle before the sweep archives it.
pub const IDLE_MS: u64 = 24 * 60 * 60 * 1000;

/// How often the sweep runs.
const SWEEP_EVERY: Duration = Duration::from_secs(60 * 60);

/// Distil one worker's conversation to text, by its `conv_id`.
pub type Distil<'a> = &'a dyn Fn(&str) -> Result<(), String>;

/// Whether `w` still holds its whole conversation: waiting on Claude, or
/// finished and kept to be continued.
pub fn unarchived(w: &Worker) -> bool {
    (w.state.live() && !w.state.working()) || w.resumable
}

/// When `w` last did anything.
fn last_active(w: &Worker) -> u64 {
    w.ended_ms.unwrap_or(0).max(w.beat_ms).max(w.turn_ms)
}

/// Archive the worker `w`, which is not working, `why` it goes: a worker
/// waiting on Claude ends as an error saying so, its question kept; a finished
/// one keeps its answer. The record to keep.
fn archive_worker(distil: Distil, mut w: Worker, why: &str, now: u64) -> Result<Worker, String> {
    distil(&w.conv_id).map_err(|e| format!("{} could not be archived: {e}", w.path()))?;
    if w.state.live() {
        w.error = Some(format!(
            "archived {why} while {}",
            if w.state == State::Question {
                "waiting on an answer"
            } else {
                "stuck"
            }
        ));
        w.state = State::Error;
        w.ended_ms = Some(now);
    }
    w.resumable = false;
    Ok(w)
}

/// Archive the thread `t` of `owner` and every worker on it still whole,
/// entering `reason` on its journal; refused while any worker on it runs. The
/// line entered.
pub fn archive_thread(
    records: &dyn Records,
    distil: Distil,
    owner: &str,
    t: Thread,
    reason: &str,
    now: u64,
) -> Result<String, String> {
    let workers: Vec<Worker> = threads::workers(records, owner, &t.name)
        .into_iter()
        .map(|w| w.observed(now))
        .collect();
    let running: Vec<String> = workers
        .iter()
        .filter(|w| w.state.working())
        .map(|w| w.path())
        .collect();
    if !running.is_empty() {
        return Err(format!(
            "workers on T:{} are still running: {}. Wait for them, then archive",
            t.name,
            running.join(", ")
        ));
    }
    let mut closed = Vec::new();
    for w in workers.into_iter().filter(unarchived) {
        let waiting = w.state.live();
        let w = archive_worker(distil, w, "with its thread", now)?;
        threads::put_worker(records, owner, &w).map_err(|e| e.to_string())?;
        if waiting {
            closed.push(w.path());
        }
    }
    let name = t.name.clone();
    let archived = Thread {
        archived: true,
        ..t
    };
    threads::put_thread(records, owner, &archived).map_err(|e| e.to_string())?;
    let text = if closed.is_empty() {
        reason.to_string()
    } else {
        format!(
            "{reason}; archived with it, unanswered: {}",
            closed.join(", ")
        )
    };
    let note = Note {
        at_ms: now,
        label: "Thread archived:".into(),
        text: text.clone(),
        worker: None,
    };
    threads::add_note(records, owner, &name, &note).map_err(|e| e.to_string())?;
    Ok(text)
}

/// One pass of the sweep at `now`: every worker idle [`IDLE_MS`] archived,
/// then every open thread whose newest activity is that old with nothing
/// running on it. How many workers and threads it archived.
pub fn sweep(records: &dyn Records, distil: Distil, now: u64) -> (usize, usize) {
    let idle = |at: u64| now.saturating_sub(at) > IDLE_MS;
    let (mut workers_archived, mut threads_archived) = (0, 0);
    for (owner, t) in threads::every_thread(records) {
        let workers: Vec<Worker> = threads::workers(records, &owner, &t.name)
            .into_iter()
            .map(|w| w.observed(now))
            .collect();
        for w in workers
            .iter()
            .filter(|w| unarchived(w) && idle(last_active(w)))
        {
            match archive_worker(distil, w.clone(), "after a day idle", now) {
                Ok(w) => match threads::put_worker(records, &owner, &w) {
                    Ok(()) => workers_archived += 1,
                    Err(e) => {
                        tracing::warn!(worker = %w.path(), "mcp: an idle worker's record could not be kept: {e}")
                    }
                },
                Err(e) => tracing::warn!("mcp: {e}"),
            }
        }
        if t.archived || workers.iter().any(|w| w.state.working()) {
            continue;
        }
        let newest = threads::notes(records, &owner, &t.name)
            .iter()
            .map(|n| n.at_ms)
            .chain(workers.iter().map(last_active))
            .fold(t.created_ms, u64::max);
        if !idle(newest) {
            continue;
        }
        let name = t.name.clone();
        match archive_thread(records, distil, &owner, t, "a day idle", now) {
            Ok(_) => threads_archived += 1,
            Err(e) => {
                tracing::warn!(thread = %name, "mcp: an idle thread could not be archived: {e}")
            }
        }
    }
    (workers_archived, threads_archived)
}

/// What the session's own archive does, as the sweep and `thread_archive`
/// call it: nothing to do before the model is loaded.
pub fn distil_with(session: &ZendSession) -> impl Fn(&str) -> Result<(), String> + '_ {
    |conv_id| match session.archive_worker(conv_id) {
        Some(Err(e)) => Err(e.to_string()),
        Some(Ok(())) | None => Ok(()),
    }
}

/// Run the sweep every [`SWEEP_EVERY`] for as long as the session lives.
pub async fn sweeping(session: Weak<ZendSession>) {
    let mut every = tokio::time::interval(SWEEP_EVERY);
    loop {
        every.tick().await;
        let Some(session) = session.upgrade() else {
            return;
        };
        // Archiving takes the engine's lock: off the runtime's threads.
        let swept = tokio::task::spawn_blocking(move || {
            let records = session.shared_substrate()?;
            Some(sweep(&records, &distil_with(&session), now_ms()))
        })
        .await;
        match swept {
            Ok(Some((0, 0))) | Ok(None) => {}
            Ok(Some((workers, threads))) => {
                tracing::info!(workers, threads, "mcp: archived what sat idle a day")
            }
            Err(e) => tracing::warn!("mcp: the idle sweep failed: {e}"),
        }
    }
}

/// Start [`sweeping`] on `runtime` for `session`.
pub fn start_sweeping(session: &Arc<ZendSession>, runtime: &Handle) {
    runtime.spawn(sweeping(Arc::downgrade(session)));
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::mcp::records::MemRecords;
    use web::auth::Role;

    const DAY: u64 = IDLE_MS;

    fn worker(name: &str, state: State, at: u64) -> Worker {
        Worker {
            thread: "review".into(),
            name: name.into(),
            conv_id: format!("c-{name}"),
            prompt: "p".into(),
            state,
            role: Role::User,
            priority: Default::default(),
            effort: Default::default(),
            started_ms: at,
            turn_ms: at,
            beat_ms: at,
            ended_ms: (!state.live()).then_some(at),
            entries_after: None,
            entries: Vec::new(),
            final_text: "ok".into(),
            error: None,
            question: None,
            activity: None,
            no_entry: None,
            stopped: false,
            resumable: state == State::Done,
            audit: None,
        }
    }

    fn thread(records: &MemRecords, workers: &[Worker]) {
        let t = Thread {
            name: "review".into(),
            created_ms: 0,
            archived: false,
        };
        threads::put_thread(records, "o", &t).unwrap();
        for w in workers {
            threads::put_worker(records, "o", w).unwrap();
        }
    }

    /// **A day idle archives the worker, and a day quiet the thread**: a
    /// finished worker kept to be continued is distilled and no longer
    /// resumable; one waiting on Claude ends as an error saying why; the
    /// thread is archived once nothing on it has moved for a day.
    #[test]
    fn a_day_idle_archives_the_workers_and_then_the_thread() {
        let records = MemRecords::default();
        thread(
            &records,
            &[
                worker("done", State::Done, 0),
                worker("asks", State::Question, 0),
                worker("old", State::Error, 0),
            ],
        );
        let distilled = RefCell::new(Vec::new());
        let distil = |c: &str| -> Result<(), String> {
            distilled.borrow_mut().push(c.to_string());
            Ok(())
        };
        assert_eq!(
            sweep(&records, &distil, DAY),
            (0, 0),
            "a day exactly is not past it"
        );
        assert_eq!(sweep(&records, &distil, DAY + 1), (2, 1));
        assert_eq!(*distilled.borrow(), ["c-asks", "c-done"]);
        let done = threads::worker(&records, "o", "review", "done").unwrap();
        assert!(!done.resumable && done.state == State::Done);
        let asks = threads::worker(&records, "o", "review", "asks").unwrap();
        assert_eq!(asks.state, State::Error);
        assert_eq!(
            asks.error.as_deref(),
            Some("archived after a day idle while waiting on an answer")
        );
        assert!(threads::thread(&records, "o", "review").unwrap().archived);
        let notes = threads::notes(&records, "o", "review");
        assert_eq!(notes.last().unwrap().text, "a day idle");
        assert_eq!(
            sweep(&records, &distil, 2 * DAY),
            (0, 0),
            "nothing is archived twice"
        );
    }

    /// **A running worker keeps its thread open**, and a thread that moved
    /// within the day is left alone, though a worker on it idle a day is not.
    #[test]
    fn a_thread_with_anything_recent_or_running_stays_open() {
        let records = MemRecords::default();
        let mut running = worker("run", State::Running, DAY);
        running.beat_ms = 2 * DAY;
        thread(&records, &[worker("done", State::Done, 0), running]);
        let distil = |_: &str| -> Result<(), String> { Ok(()) };
        assert_eq!(sweep(&records, &distil, 2 * DAY), (1, 0));
        assert!(!threads::thread(&records, "o", "review").unwrap().archived);

        let records = MemRecords::default();
        thread(&records, &[worker("done", State::Done, DAY)]);
        assert_eq!(sweep(&records, &distil, DAY + 2), (0, 0));
        assert!(!threads::thread(&records, "o", "review").unwrap().archived);
    }

    /// `thread_archive` is refused while a worker runs, and otherwise archives
    /// every worker still whole, naming the ones that waited on Claude.
    #[test]
    fn archiving_a_thread_names_the_unanswered_and_refuses_while_running() {
        let records = MemRecords::default();
        thread(
            &records,
            &[
                worker("done", State::Done, 5),
                worker("asks", State::Question, 5),
            ],
        );
        let distil = |_: &str| -> Result<(), String> { Ok(()) };
        let t = threads::thread(&records, "o", "review").unwrap();
        assert_eq!(
            archive_thread(&records, &distil, "o", t, "the feature is done", 9).unwrap(),
            "the feature is done; archived with it, unanswered: T:review/asks"
        );
        assert!(
            !threads::worker(&records, "o", "review", "done")
                .unwrap()
                .resumable
        );

        let records = MemRecords::default();
        thread(&records, &[worker("run", State::Running, 5)]);
        let t = threads::thread(&records, "o", "review").unwrap();
        let err = archive_thread(&records, &distil, "o", t, "done", 9).unwrap_err();
        assert!(
            err.starts_with("workers on T:review are still running: T:review/run"),
            "{err}"
        );
    }
}
