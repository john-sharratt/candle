//! Writes a task's journal lines and its record while the turn runs, so a reader
//! sees that the task is still running and what it has done.
//!
//! A beat rewrites the record while the turn runs, carrying the steps taken and
//! the entries drafted so far, and it wakes every waiter. It is the only writer of
//! the record during a turn, and it stops before the settled record is written.
//! Steps that come quicker than the gap are written together. A member that stops
//! a worker as it leaves records that the worker is handed on, so another member
//! resumes it at once. The worker's state stays running, because nothing ended it.
//! A chat's thread keeps no journal, so nothing is entered for it.

use super::*;

/// Enter a line about `worker` on its thread's journal — none for Zen's side
/// of a chat, whose thread keeps no copy of what was said
/// (`docs/zen_chat.md` §2).
pub(super) fn note(
    records: &dyn Records,
    owner: &str,
    worker: &Worker,
    label: &str,
    text: String,
) -> Result<(), String> {
    if is_chat_thread(&worker.thread) {
        return Ok(());
    }
    threads::add_note(
        records,
        owner,
        &worker.thread,
        &Note {
            at_ms: now_ms(),
            label: label.into(),
            text: format!("{} — {text}", worker.path()),
            worker: Some(worker.path()),
            client: None,
        },
    )
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// Say on `worker`'s record that this member stopped it as it left: no runner,
/// handed now, so another member resumes it at once (`docs/turn_handover.md`
/// §5). Its state stays running — nothing ended it.
pub(super) fn hand_left(records: &dyn Records, owner: &str, worker: &Worker) {
    if let Some(mut w) = threads::worker(records, owner, &worker.thread, &worker.name) {
        w.runner = None;
        w.handed_ms = Some(now_ms());
        if let Err(e) = threads::put_worker(records, owner, &w) {
            tracing::warn!(conv_id = %w.conv_id, "mcp: the worker could not be marked handed: {e}");
        }
    }
}

/// Say on `worker`'s record that it still runs, now.
pub(super) fn beat_now(records: &dyn Records, owner: &str, worker: &Worker) {
    if let Some(mut w) = threads::worker(records, owner, &worker.thread, &worker.name) {
        w.beat_ms = now_ms();
        if let Err(e) = threads::put_worker(records, owner, &w) {
            tracing::warn!(conv_id = %w.conv_id, "mcp: the task's beat could not be kept: {e}");
        }
    }
}

/// What a running turn has done, shared between the turn that counts it and
/// the beat that writes it.
#[derive(Clone, Default)]
pub(super) struct Live {
    pub(super) activity: Arc<Mutex<Activity>>,
    /// Woken when a step is worth telling at once.
    pub(super) moved: Arc<Notify>,
}

impl Live {
    pub(super) fn observe(&self, item: &StreamItem) {
        let step = self
            .activity
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .observe_at(item, now_ms());
        if step {
            self.moved.notify_one();
        }
    }

    pub(super) fn set_doing(&self, doing: Doing) {
        self.activity.lock().unwrap_or_else(|e| e.into_inner()).now = doing;
        self.moved.notify_one();
    }

    pub(super) fn snapshot(&self) -> Activity {
        self.activity
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

/// Say on the worker's record that its task still runs, what its turn has
/// done, and the journal entries drafted so far — every [`BEAT_EVERY_MS`], and
/// at once when a round ends or a batch of tool calls begins — then wake every
/// waiter; until aborted. The only writer of the record while the turn runs.
pub(super) async fn beat(
    session: Arc<ZendSession>,
    records: SharedSubstrate,
    owner: String,
    thread: String,
    name: String,
    live: Live,
) {
    let mut every = tokio::time::interval(Duration::from_millis(BEAT_EVERY_MS));
    every.tick().await;
    let mut kept = Instant::now();
    loop {
        tokio::select! {
            _ = every.tick() => {}
            _ = live.moved.notified() => {}
        }
        // Steps that come quicker than the gap are told together: the record
        // replicates, and every write of it is kept.
        let since = kept.elapsed();
        if since < STEP_GAP {
            tokio::time::sleep(STEP_GAP - since).await;
        }
        kept = Instant::now();
        let Some(mut w) = threads::worker(&records, &owner, &thread, &name) else {
            return;
        };
        // Another member took it over: the record is that member's to keep.
        if worker_hop::runs_elsewhere(true, w.runner, worker_hop::own_node()) {
            return;
        }
        w.beat_ms = now_ms();
        w.activity = Some(live.snapshot());
        w.entries = entries_after(session.journal(&w.conv_id).as_ref(), w.entries_after);
        if let Err(e) = threads::put_worker(&records, &owner, &w) {
            tracing::warn!(conv_id = %w.conv_id, "mcp: the task's beat could not be kept: {e}");
        }
        session.mcp().bump();
    }
}
