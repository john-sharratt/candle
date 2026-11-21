//! Starts a worker's turn on its first message, or on a round handed to it, under
//! the lease on its conversation's turns.
//!
//! A turn begins from one of three things: a message, a round handed here by a
//! slower member, or a round a restart cut off, which may have run some of its
//! calls. The lease on the conversation's turns is waited for up to `LEASE_WAIT`.
//! A worker whose lease cannot be had is given up, and a chat is told so. The record
//! is written before the turn makes the conversation's slot, which reads it, so the
//! turn is built from this member's current view of the conversation.

use super::*;

/// Keep `worker` as running and start its turn on `message`.
pub(in super::super) fn begin(
    session: &Arc<ZendSession>,
    records: &dyn Records,
    owner: &str,
    worker: Worker,
    message: String,
) -> Result<Worker, String> {
    begin_held(
        session,
        records,
        owner,
        worker,
        First::Message(message),
        None,
    )
}

/// Keep `worker` as running and start its turn from `first`, under `lease` when
/// the caller already took it.
pub(super) fn begin_held(
    session: &Arc<ZendSession>,
    records: &dyn Records,
    owner: &str,
    worker: Worker,
    first: First,
    lease: Option<Guard>,
) -> Result<Worker, String> {
    // A stop sent as the last turn ended — after its settling looked — is
    // not this turn's.
    stops::take(records, &worker.conv_id);
    // Its last turns may have run on another member, after what this one
    // keeps of the conversation in memory: the turn here is built afresh.
    let mut worker = worker;
    let here = worker_hop::own_node();
    if worker.runner != here {
        session.forget(&worker.conv_id);
    }
    worker.runner = here;
    threads::put_worker(records, owner, &worker).map_err(|e| e.to_string())?;
    // Recorded before the turn makes the conversation's slot, which reads it.
    if let Some(Err(e)) = session.set_decode_band(&worker.conv_id, worker.priority.band()) {
        tracing::warn!(conv_id = %worker.conv_id, "mcp: the task's priority could not be kept: {e}");
    }
    session.mcp().bump();
    tokio::spawn(run(
        Arc::clone(session),
        owner.to_string(),
        worker.clone(),
        first,
        lease,
    ));
    Ok(worker)
}

/// What a worker's next turn starts from.
pub(super) enum First {
    /// A message: the task, an answer, a correction.
    Message(String),
    /// A round whose call turn `call_turn`, round `iteration` of its turn, is
    /// sealed and whose tools have not run — handed here by a slower member,
    /// or kept when a handing failed.
    Round { call_turn: u32, iteration: usize },
    /// A round whose call turn `call_turn` is sealed and which a restart cut
    /// off: some of its tools may have run.
    Interrupted { call_turn: u32 },
}

impl First {
    /// What the turn is sent, and the round it finishes.
    pub(super) fn into_start(self) -> (String, Option<Resume>) {
        match self {
            First::Message(message) => (message, None),
            First::Round {
                call_turn,
                iteration,
            } => (
                String::new(),
                Some(Resume {
                    call_turn,
                    dispatch: Dispatch::Live,
                    iteration,
                }),
            ),
            First::Interrupted { call_turn } => (
                String::new(),
                Some(Resume {
                    call_turn,
                    dispatch: Dispatch::Resumed,
                    iteration: 0,
                }),
            ),
        }
    }
}

/// Take the lease on `conv_id`'s turns, waiting up to [`LEASE_WAIT`] for
/// whoever holds it.
pub(super) async fn hold_turns(conv_id: &str) -> Option<Guard> {
    let until = Instant::now() + LEASE_WAIT;
    loop {
        let conv_id = conv_id.to_string();
        let share = tokio::task::spawn_blocking(move || ingest_lease::share_worker(&conv_id))
            .await
            .ok()?;
        if let Share::Mine(lease) = share {
            return Some(lease);
        }
        if Instant::now() >= until {
            return None;
        }
        tokio::time::sleep(LEASE_LOOK).await;
    }
}

/// End `worker`, whose turns this member could not get the lease on.
pub(super) fn give_up(
    session: &Arc<ZendSession>,
    records: &dyn Records,
    owner: &str,
    worker: &mut Worker,
) {
    let why = "another member holds this task's turns, so it was not started here";
    worker.state = State::Error;
    worker.error = Some(why.into());
    worker.ended_ms = Some(now_ms());
    // A chat hears of it as of any answer that failed, and one closed while
    // it waited is disposed of now.
    let chat = worker.kind.is_chat();
    if chat {
        talk::answered(records, owner, worker);
    }
    if let Err(e) = threads::put_worker(records, owner, worker) {
        tracing::warn!(conv_id = %worker.conv_id, "mcp: the worker's outcome could not be kept: {e}");
    }
    if chat {
        talk::settled(session, owner, worker);
    }
    if let Err(e) = note(
        records,
        owner,
        worker,
        "Task ended:",
        format!("error: {why}"),
    ) {
        tracing::warn!(conv_id = %worker.conv_id, "mcp: the outcome could not be entered on the thread: {e}");
    }
}
