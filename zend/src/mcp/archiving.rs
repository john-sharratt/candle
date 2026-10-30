//! Archiving threads and their workers (`docs/mcp_guides.md` §3): by
//! `thread_archive`, and by the sweep once they have sat idle a day.
//!
//! A worker that finished its task is kept whole — its conversation and its
//! KV — so Claude can continue it with `repo_task_reply`, as it can a worker that
//! asked or is stuck. Archiving distils the conversation to text
//! (`ZendSession::archive_worker`), which is one way: an archived worker's
//! evidence stays readable, and the worker is never continued.
//!
//! The sweep archives a worker nothing has touched for [`IDLE_MS`], and a
//! thread whose newest activity is that old with nothing running on it. It
//! touches only these workers: a person's own zend conversations are never
//! archived but by that person.
//!
//! A worker is archived in three steps: read again and claimed under the
//! lock a reply takes ([`workers::holding`]) — a reply refuses a claimed
//! worker — then distilled outside it, then, under it again, its record
//! written as archived only if the distil succeeded and nothing changed the
//! record meanwhile. A failed distil writes nothing, so no record ever says
//! archived of a conversation that was not, and the next sweep tries again.
//!
//! Every member sweeps. Two members sweeping in the moment before one's
//! write replicates to the other can each archive one thread: distilling
//! twice is harmless, and the thread's journal gets the line twice.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::Duration;

use tokio::runtime::Handle;

use super::records::Records;
use super::threads::{self, Note, State, Thread, Worker};
use super::workers::{self, now_ms};
use crate::session::ZendSession;

/// How long a worker or a thread sits idle before the sweep archives it.
pub const IDLE_MS: u64 = 24 * 60 * 60 * 1000;

/// How long a lost worker's beat has been silent before the sweep archives
/// it. "Lost" is read off a record that replicates: a member whose copy lags,
/// or whose runtime stalls, sees a live worker's beat stop for a while, and
/// archiving distils a conversation for good — so only a silence no lag or
/// stall explains counts.
pub const LOST_ARCHIVE_MS: u64 = 60 * 60 * 1000;

/// How often the sweep runs.
const SWEEP_EVERY: Duration = Duration::from_secs(60 * 60);

/// Distil one worker's conversation to text, by its `conv_id`.
pub type Distil<'a> = &'a dyn Fn(&str) -> Result<(), String>;

/// Whether `w` — as [`Worker::observed`] reads it — was lost: its member
/// stopped running it, so its record still says it runs and nothing ended it.
pub fn lost(w: &Worker) -> bool {
    w.state == State::Error && w.ended_ms.is_none()
}

/// Whether `w` still holds its whole conversation: waiting on Claude,
/// finished and kept to be continued, or lost with nothing to end it.
pub fn unarchived(w: &Worker) -> bool {
    (w.state.live() && !w.state.working()) || w.resumable || lost(w)
}

/// When `w` last did anything.
fn last_active(w: &Worker) -> u64 {
    w.ended_ms.unwrap_or(0).max(w.beat_ms).max(w.turn_ms)
}

/// The newest thing that happened on the thread `t`: its birth, an attach —
/// counted to the end of its mark's hour, since a later attach within it
/// writes nothing — a journal line, or a worker's last activity.
fn newest(records: &dyn Records, owner: &str, t: &Thread, workers: &[Worker]) -> u64 {
    let attached = match t.attached_ms {
        0 => 0,
        at => at.saturating_add(threads::ATTACH_MARK_MS),
    };
    threads::notes(records, owner, &t.name)
        .iter()
        .map(|n| n.at_ms)
        .chain(workers.iter().map(last_active))
        .fold(t.created_ms.max(attached), u64::max)
}

/// The record of the worker `w`, which is not working, once archived `why`:
/// a worker waiting on Claude ends as an error saying so, its question kept,
/// ended when it last did anything — archiving is not activity; a finished
/// one keeps its answer, and can no longer be continued.
fn archived(mut w: Worker, why: &str) -> Worker {
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
        w.ended_ms = Some(last_active(&w));
    } else if lost(&w) {
        w.ended_ms = Some(last_active(&w));
    }
    w.resumable = false;
    w
}

/// The conversations being archived on this member: claimed under the lock
/// a reply takes, distilled outside it, released once their records say
/// archived. A reply refuses a worker while it is here ([`being_archived`]).
static ARCHIVING: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());

fn archiving() -> MutexGuard<'static, BTreeSet<String>> {
    ARCHIVING.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether the conversation `conv_id` is being archived on this member.
pub fn being_archived(conv_id: &str) -> bool {
    archiving().contains(conv_id)
}

/// Distil each claimed worker — as its record read, observed at `now`, when
/// claimed — outside the lock, since distilling takes the engine's; then,
/// under the lock, keep the archived record of each that distilled, when
/// nothing has changed its record since, and release every claim. Nothing is
/// written for one that failed, so no record says archived of a conversation
/// that was not. A task that was lost has its end entered on its thread —
/// nothing else ever would. The paths that failed, each with why.
fn distil_claimed(
    records: &dyn Records,
    distil: Distil,
    owner: &str,
    claimed: Vec<Worker>,
    (why, now): (&str, u64),
) -> Vec<String> {
    let outcomes: Vec<(Worker, Result<(), String>)> = claimed
        .into_iter()
        .map(|w| {
            let distilled = distil(&w.conv_id);
            (w, distilled)
        })
        .collect();
    let _held = workers::holding();
    let mut failed = Vec::new();
    for (before, distilled) in outcomes {
        archiving().remove(&before.conv_id);
        if let Err(e) = distilled {
            failed.push(format!("{} ({e})", before.path()));
            continue;
        }
        let current =
            threads::worker(records, owner, &before.thread, &before.name).map(|w| w.observed(now));
        if current.as_ref() != Some(&before) {
            tracing::warn!(worker = %before.path(), "mcp: a worker changed while it was archived; its record is left as it is");
            continue;
        }
        if let Err(e) = threads::put_worker(records, owner, &archived(before.clone(), why)) {
            failed.push(format!(
                "{} (its record could not be kept: {e})",
                before.path()
            ));
            continue;
        }
        if lost(&before) && !before.is_part() {
            let note = Note {
                at_ms: now,
                label: "Task ended:".into(),
                text: format!(
                    "{} — lost: the member running it restarted, and its turn with it",
                    before.path()
                ),
                worker: Some(before.path()),
                client: None,
            };
            if let Err(e) = threads::add_note(records, owner, &before.thread, &note) {
                tracing::warn!(worker = %before.path(), "mcp: a lost task's end could not be entered: {e}");
            }
        }
    }
    failed
}

/// Claim `w` for archiving, under the lock: false when it is claimed already.
fn claim(w: &Worker) -> bool {
    archiving().insert(w.conv_id.clone())
}

/// How archiving a thread came out.
#[derive(Debug, PartialEq, Eq)]
pub enum Archived {
    /// Archived: the line entered on its journal, and any worker whose
    /// conversation could not be distilled — left whole, to be archived again.
    Done { line: String, failed: Vec<String> },
    /// It was archived already; any worker on it still whole is archived now,
    /// and those that could not be are named.
    Already { failed: Vec<String> },
    /// Something happened on it after the sweep's cutoff.
    Active,
}

/// When a thread may be archived: always, or only when nothing on it
/// happened at or after a cutoff — the sweep's idle test, made again on what
/// the lock lets it read.
#[derive(Debug, Clone, Copy)]
pub enum When {
    Now,
    QuietSince(u64),
}

/// Archive the thread `name` of `owner` and every worker on it still whole,
/// entering `reason` on its journal; refused while any worker on it runs.
pub fn archive_thread(
    records: &dyn Records,
    distil: Distil,
    owner: &str,
    name: &str,
    reason: &str,
    when: When,
    now: u64,
) -> Result<Archived, String> {
    let (line, claimed) = {
        // Held across the reads, the claims and the thread's record — not
        // the distils — so no attach or reply lands between the thread's
        // being read and its saying archived.
        let _held = workers::holding();
        let Some(t) = threads::thread(records, owner, name) else {
            return Err(format!("there is no thread `T:{name}`"));
        };
        let workers: Vec<Worker> = threads::workers(records, owner, name)
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
                "workers on T:{name} are still running: {}. Wait for them, then archive",
                running.join(", ")
            ));
        }
        if let When::QuietSince(cut) = when {
            if t.archived || newest(records, owner, &t, &workers) >= cut {
                return Ok(Archived::Active);
            }
        }
        let whole: Vec<Worker> = workers.into_iter().filter(unarchived).collect();
        let closed: Vec<String> = whole
            .iter()
            .filter(|w| w.state.live())
            .map(Worker::path)
            .collect();
        let claimed: Vec<Worker> = whole.into_iter().filter(claim).collect();
        let line = if t.archived {
            None
        } else {
            let archived = Thread {
                archived: true,
                ..t
            };
            let line = if closed.is_empty() {
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
                text: line.clone(),
                worker: None,
                client: None,
            };
            let kept = threads::put_thread(records, owner, &archived)
                .and_then(|()| threads::add_note(records, owner, name, &note));
            if let Err(e) = kept {
                let mut held = archiving();
                for w in &claimed {
                    held.remove(&w.conv_id);
                }
                return Err(e.to_string());
            }
            Some(line)
        };
        (line, claimed)
    };
    let failed = distil_claimed(records, distil, owner, claimed, ("with its thread", now));
    Ok(match line {
        Some(line) => Archived::Done { line, failed },
        None => Archived::Already { failed },
    })
}

/// One pass of the sweep at `now`: every worker idle [`IDLE_MS`] archived,
/// then every open thread whose newest activity is that old with nothing
/// running on it. How many workers and threads it archived.
pub fn sweep(records: &dyn Records, distil: Distil, now: u64) -> (usize, usize) {
    let idle = |at: u64| now.saturating_sub(at) > IDLE_MS;
    // A lost worker is archived once its beat has been silent an hour:
    // nothing can continue it, and until then its record says it runs.
    let silent = |at: u64| now.saturating_sub(at) > LOST_ARCHIVE_MS;
    let due =
        |w: &Worker| unarchived(w) && ((lost(w) && silent(last_active(w))) || idle(last_active(w)));
    let (mut workers_archived, mut threads_archived) = (0, 0);
    for (owner, t) in threads::every_thread(records) {
        let workers: Vec<Worker> = threads::workers(records, &owner, &t.name)
            .into_iter()
            .map(|w| w.observed(now))
            .collect();
        let mut stuck = false;
        for w in workers.iter().filter(|w| due(w)) {
            // Read again and claimed under the lock a reply takes, so a
            // worker Claude continued a moment ago is never archived under
            // its turn, and one Claude tries to continue now is refused.
            let claimed = {
                let _held = workers::holding();
                let Some(w) = threads::worker(records, &owner, &t.name, &w.name)
                    .map(|w| w.observed(now))
                    .filter(|w| due(w))
                    .filter(claim)
                else {
                    continue;
                };
                w
            };
            match distil_claimed(
                records,
                distil,
                &owner,
                vec![claimed],
                ("after a day idle", now),
            )
            .as_slice()
            {
                [] => workers_archived += 1,
                failed => {
                    // Left whole; the next sweep tries again, and this one
                    // leaves its thread for then.
                    tracing::warn!(
                        "mcp: an idle worker could not be archived: {}",
                        failed.join("; ")
                    );
                    stuck = true;
                }
            }
        }
        if stuck || t.archived || workers.iter().any(|w| w.state.working()) {
            continue;
        }
        if !idle(newest(records, &owner, &t, &workers)) {
            continue;
        }
        let cut = now.saturating_sub(IDLE_MS);
        match archive_thread(
            records,
            distil,
            &owner,
            &t.name,
            "a day idle",
            When::QuietSince(cut),
            now,
        ) {
            Ok(Archived::Done { failed, .. }) => {
                threads_archived += 1;
                if !failed.is_empty() {
                    tracing::warn!(thread = %t.name, "mcp: workers of an idle thread could not be archived: {}", failed.join("; "));
                }
            }
            Ok(Archived::Already { .. } | Archived::Active) => {}
            Err(e) => {
                tracing::warn!(thread = %t.name, "mcp: an idle thread could not be archived: {e}")
            }
        }
    }
    (workers_archived, threads_archived)
}

/// What the session's own archive does, as the sweep and `thread_archive`
/// call it: refused before the model is loaded, so no record says archived of
/// a conversation that was not.
pub fn distil_with(session: &ZendSession) -> impl Fn(&str) -> Result<(), String> + '_ {
    |conv_id| match session.archive_worker(conv_id) {
        Some(Err(e)) => Err(e.to_string()),
        Some(Ok(())) => Ok(()),
        None => Err("the model is still loading".into()),
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
        if !session.is_loaded() {
            continue;
        }
        // Archiving writes through the persistence owner, whose flow control
        // can wait for room: off the runtime's threads.
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::mcp::records::MemRecords;
    use web::auth::Role;

    const DAY: u64 = IDLE_MS;

    /// Claims are per process, and the tests run at once: each worker's
    /// conversation is its own, `c-<name>-<n>`.
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    /// A conversation id without its test's number.
    fn conv_name(conv_id: &str) -> &str {
        conv_id.rsplit_once('-').map_or(conv_id, |(name, _)| name)
    }

    fn worker(name: &str, state: State, at: u64) -> Worker {
        Worker {
            thread: "review".into(),
            name: name.into(),
            conv_id: format!("c-{name}-{}", NEXT.fetch_add(1, Ordering::Relaxed)),
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
            checked: false,
        }
    }

    fn thread(records: &MemRecords, workers: &[Worker]) {
        let t = Thread {
            name: "review".into(),
            created_ms: 0,
            archived: false,
            attached_ms: 0,
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
        let names: Vec<String> = distilled
            .borrow()
            .iter()
            .map(|c| conv_name(c).to_string())
            .collect();
        assert_eq!(names, ["c-asks", "c-done"]);
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

    /// **Attaching again is activity**: a swept thread Claude reopens is not
    /// swept again while the attach is within the day.
    #[test]
    fn a_reopened_thread_is_not_swept_again() {
        let records = MemRecords::default();
        thread(&records, &[]);
        let distil = |_: &str| -> Result<(), String> { Ok(()) };
        assert_eq!(sweep(&records, &distil, DAY + 1), (0, 1));
        // Its newest note — the archive's — is now a day old.
        threads::attach(&records, "o", "review", 2 * DAY).unwrap();
        assert_eq!(sweep(&records, &distil, 2 * DAY + 2), (0, 0));
        assert!(!threads::thread(&records, "o", "review").unwrap().archived);
        assert_eq!(
            sweep(&records, &distil, 3 * DAY + 1),
            (0, 0),
            "an attach counts to the end of its mark's hour"
        );
        let after_mark = 3 * DAY + threads::ATTACH_MARK_MS + 1;
        assert_eq!(sweep(&records, &distil, after_mark), (0, 1));
    }

    /// **A distil that fails writes nothing**: no record says archived of a
    /// conversation that was not, its claim is released, its thread is left
    /// for the next sweep, and that sweep archives both.
    #[test]
    fn a_failed_distil_leaves_the_worker_and_its_thread_as_they_were() {
        let records = MemRecords::default();
        let done = worker("done", State::Done, 0);
        thread(&records, std::slice::from_ref(&done));
        let refuse = |_: &str| -> Result<(), String> { Err("the model is still loading".into()) };
        let now = DAY + 1;
        assert_eq!(sweep(&records, &refuse, now), (0, 0));
        assert_eq!(
            threads::worker(&records, "o", "review", "done").unwrap(),
            done
        );
        assert!(!being_archived(&done.conv_id));
        assert!(!threads::thread(&records, "o", "review").unwrap().archived);
        let ok = |_: &str| -> Result<(), String> { Ok(()) };
        assert_eq!(sweep(&records, &ok, now + 1), (1, 1));
        assert!(
            !threads::worker(&records, "o", "review", "done")
                .unwrap()
                .resumable
        );
    }

    /// **A worker being archived is refused to everyone else**: a second
    /// claim fails until the first is released, and an archive that finds
    /// the worker changed under it leaves the record alone.
    #[test]
    fn a_claimed_worker_is_claimed_once_and_a_changed_one_is_left() {
        let records = MemRecords::default();
        let done = worker("done", State::Done, 0);
        thread(&records, std::slice::from_ref(&done));
        assert!(claim(&done) && !claim(&done) && being_archived(&done.conv_id));
        let mut continued = done.clone();
        continued.state = State::Running;
        threads::put_worker(&records, "o", &continued).unwrap();
        let ok = |_: &str| -> Result<(), String> { Ok(()) };
        assert!(distil_claimed(&records, &ok, "o", vec![done.clone()], ("x", 0)).is_empty());
        assert!(!being_archived(&done.conv_id));
        assert_eq!(
            threads::worker(&records, "o", "review", "done").unwrap(),
            continued
        );
    }

    /// **A lost worker is archived once its beat has been silent an hour,
    /// and its end entered**: its record said it ran, nothing could continue
    /// it, and nothing else would ever say it ended. A shorter silence — a
    /// copy that lags, a stalled runtime — is left alone.
    #[test]
    fn a_lost_worker_is_archived_after_an_hours_silence_with_its_end_on_the_thread() {
        let records = MemRecords::default();
        let mut gone = worker("gone", State::Running, 0);
        gone.beat_ms = 10;
        thread(&records, std::slice::from_ref(&gone));
        let distil = |_: &str| -> Result<(), String> { Ok(()) };
        assert_eq!(
            sweep(&records, &distil, 10 + threads::BEAT_LOST_MS + 1),
            (0, 0),
            "lost to a reader, but not yet silent long enough to archive"
        );
        let now = 10 + LOST_ARCHIVE_MS + 1;
        assert_eq!(sweep(&records, &distil, now), (1, 0));
        let after = threads::worker(&records, "o", "review", "gone").unwrap();
        assert_eq!(
            (after.state, after.ended_ms, after.resumable),
            (State::Error, Some(10), false)
        );
        let notes = threads::notes(&records, "o", "review");
        assert_eq!(
            notes.last().unwrap().text,
            "T:review/gone — lost: the member running it restarted, and its turn with it"
        );
        assert_eq!(sweep(&records, &distil, now + 1), (0, 0), "once only");
    }

    /// **The sweep's idle test is made again under the lock**: a thread
    /// with something after the cutoff is left open. Archiving one archived
    /// already says so, and archives any worker left whole on it.
    #[test]
    fn a_thread_active_since_the_cutoff_or_already_archived_is_left() {
        let records = MemRecords::default();
        thread(&records, &[worker("done", State::Done, 50)]);
        let ok = |_: &str| -> Result<(), String> { Ok(()) };
        let refuse = |_: &str| -> Result<(), String> { Err("busy".into()) };
        let quiet = When::QuietSince(50);
        assert_eq!(
            archive_thread(&records, &ok, "o", "review", "a day idle", quiet, 99),
            Ok(Archived::Active)
        );
        assert!(matches!(
            archive_thread(&records, &refuse, "o", "review", "done", When::Now, 99),
            Ok(Archived::Done { failed, .. }) if failed.len() == 1
        ));
        assert!(
            threads::worker(&records, "o", "review", "done")
                .unwrap()
                .resumable
        );
        assert_eq!(
            archive_thread(&records, &ok, "o", "review", "done", When::Now, 99),
            Ok(Archived::Already { failed: Vec::new() })
        );
        assert!(
            !threads::worker(&records, "o", "review", "done")
                .unwrap()
                .resumable
        );
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
        assert_eq!(
            archive_thread(
                &records,
                &distil,
                "o",
                "review",
                "the feature is done",
                When::Now,
                9
            ),
            Ok(Archived::Done {
                line: "the feature is done; archived with it, unanswered: T:review/asks".into(),
                failed: Vec::new()
            })
        );
        assert!(
            !threads::worker(&records, "o", "review", "done")
                .unwrap()
                .resumable
        );

        let records = MemRecords::default();
        thread(&records, &[worker("run", State::Running, 5)]);
        let err =
            archive_thread(&records, &distil, "o", "review", "done", When::Now, 9).unwrap_err();
        assert!(
            err.starts_with("workers on T:review are still running: T:review/run"),
            "{err}"
        );
    }
}
