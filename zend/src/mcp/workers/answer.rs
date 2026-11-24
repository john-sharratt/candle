//! Answers a worker that asked, is stuck, or finished and was kept, by making the
//! answer its next turn in the same conversation.
//!
//! An answer reopens a worker that was kept whole, even when its thread was
//! archived while it was left. Messages sent to a running worker are queued for
//! its next tool round, and a stop takes effect at that round. A stop is ordered
//! against a turn beginning, so a stop meant for the turn before it is cleared
//! and does not end the new one.

use super::*;

/// Answer the worker `name` of `thread`, which asked, is stuck, or finished
/// and is kept to be continued: the answer is its next turn, in the same
/// conversation. The worker's record, `running` again.
pub fn answer(
    session: &Arc<ZendSession>,
    caller: &Caller,
    thread: &str,
    name: &str,
    text: &str,
) -> Result<Worker, String> {
    let records = session.shared_substrate().ok_or_else(loading)?;
    let owner = caller.key();
    if text.trim().is_empty() {
        return Err("the `answer` is empty".into());
    }
    let _held = holding();
    let Some(found) = threads::worker(&records, &owner, thread, name) else {
        return Err(format!("there is no task `task:{thread}/{name}`"));
    };
    let mut worker = found.observed(now_ms());
    if archiving::being_archived(&worker.conv_id) {
        return Err(format!(
            "`{}` is being archived, so it cannot be continued: start a new task, citing its \
             entries",
            worker.path()
        ));
    }
    let label = match worker.state {
        State::Question | State::Stuck => "Claude answered:",
        State::Done if worker.resumable => "Claude continued:",
        State::Running => {
            return Err(format!(
                "`{}` is still working: nudge it with `repo_task_nudge`, or wait",
                worker.path()
            ))
        }
        State::Done => {
            return Err(format!(
                "`{}` finished and was archived, so it cannot be continued: start a new task, \
                 citing its entries",
                worker.path()
            ))
        }
        _ => {
            return Err(format!(
                "`{}` ended and cannot be continued: start a new task",
                worker.path()
            ))
        }
    };
    let now = now_ms();
    // Continuing a worker is work on its thread: it reopens a thread archived
    // while the worker was left whole.
    threads::attach(&records, &owner, thread, now).map_err(|e| e.to_string())?;
    ready_for_turn(session, &mut worker, caller, now);
    note(
        &records,
        &owner,
        &worker,
        label,
        first_line(text, PROMPT_QUOTE),
    )?;
    begin(
        session,
        &records,
        &owner,
        worker,
        format!("{text}{ASKING}"),
        Vec::new(),
    )
}

/// `worker`, which settled, made ready for its next turn, run for `caller` —
/// as their role now — from `now`: running again, its last ending's marks
/// cleared, and its entries counted from its journal's newest.
pub(in super::super) fn ready_for_turn(
    session: &ZendSession,
    worker: &mut Worker,
    caller: &Caller,
    now: u64,
) {
    worker.state = State::Running;
    worker.role = caller.role;
    worker.person = Some(caller.person.clone());
    worker.beat_ms = now;
    worker.turn_ms = now;
    worker.ended_ms = None;
    worker.resumable = false;
    worker.audit = None;
    worker.question = None;
    worker.entries_after = newest_entry(session.journal(&worker.conv_id).as_ref());
    worker.entries = Vec::new();
}

/// Queue `text` for the running worker `name` of `thread`, taken at its next
/// tool round (`super::nudges`).
pub fn nudge(
    session: &ZendSession,
    caller: &Caller,
    thread: &str,
    name: &str,
    text: &str,
) -> Result<Worker, String> {
    let records = session.shared_substrate().ok_or_else(loading)?;
    let owner = caller.key();
    if text.trim().is_empty() {
        return Err("the `message` is empty".into());
    }
    let Some(found) = threads::worker(&records, &owner, thread, name) else {
        return Err(format!("there is no task `task:{thread}/{name}`"));
    };
    let worker = found.observed(now_ms());
    if !worker.state.working() {
        return Err(format!(
            "`{}` is not working, so it would never read a nudge; {}",
            worker.path(),
            if worker.state.live() {
                "it waits on an answer — use `repo_task_reply`"
            } else {
                "it has finished — start a new task"
            }
        ));
    }
    nudges::queue(&records, &worker.conv_id, Sender::Starter, text, now_ms())
        .map_err(|e| e.to_string())?;
    note(
        &records,
        &owner,
        &worker,
        "Claude nudged:",
        first_line(text, PROMPT_QUOTE),
    )?;
    Ok(worker)
}

/// Stop the running worker `name` of `thread` at its next tool round, for
/// `reason` (`super::stops`).
pub fn stop(
    session: &ZendSession,
    caller: &Caller,
    thread: &str,
    name: &str,
    reason: &str,
) -> Result<Worker, String> {
    let records = session.shared_substrate().ok_or_else(loading)?;
    let owner = caller.key();
    // Ordered against a turn beginning, which clears a stop meant for the
    // turn before it.
    let _held = holding();
    let Some(found) = threads::worker(&records, &owner, thread, name) else {
        return Err(format!("there is no task `task:{thread}/{name}`"));
    };
    let worker = found.observed(now_ms());
    if !worker.state.working() {
        return Err(format!(
            "`{}` is not running, so there is nothing to stop; {}",
            worker.path(),
            if worker.state.live() {
                "it waits on an answer — redirect it with `repo_task_reply`"
            } else {
                "it has finished"
            }
        ));
    }
    stops::request(&records, &worker.conv_id, reason).map_err(|e| e.to_string())?;
    let line = if reason.trim().is_empty() {
        "at its next tool round".to_string()
    } else {
        format!(
            "at its next tool round: {}",
            first_line(reason, PROMPT_QUOTE)
        )
    };
    note(&records, &owner, &worker, "Claude stopped:", line)?;
    Ok(worker)
}
