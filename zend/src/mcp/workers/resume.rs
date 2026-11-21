//! Restarts a worker that a restart cut off, with what it is told and the round it
//! had reached.
//!
//! A cut-off worker is told its whole task and what its journal says it has done,
//! since a turn's context is chosen by relevance and would otherwise lose the task.
//! A round it stopped in the middle of is finished first, with its results, because
//! a new message would leave those calls unanswered. A chat being disposed of is
//! never restarted, and a worker that is no longer lost is left alone.

use super::*;

/// What `worker`, cut off, is told: [`RESUMED`], its task in full, and what its
/// journal says it did. A turn's context is selected by relevance, so nothing
/// else keeps the task in front of it: told only to finish "the task you were
/// given", a worker 88 turns in took up something else it could see, reported
/// that done, and the task it had was never finished (2026-10-31).
pub(super) fn resumed_message(worker: &Worker) -> String {
    if worker.kind.is_chat() {
        return format!(
            "{CHAT_RESUMED}\n\nThe message you were answering:\n{}",
            worker.prompt
        );
    }
    let mut message = format!("{RESUMED}\n\nYour task:\n{}", worker.prompt);
    let skipped = worker.entries.len().saturating_sub(RESUMED_ENTRIES);
    let kept = &worker.entries[skipped..];
    if !kept.is_empty() {
        message.push_str("\n\nWhat your journal says you did so far:");
        for entry in kept {
            message.push_str("\n- ");
            message.push_str(&first_line(&entry.text, RESUMED_ENTRY_QUOTE));
        }
    }
    message
}

/// Make `worker`, cut off, run again from `now`, the try entered on its thread.
/// What it should be told is read first, off the entries this clears.
pub(super) fn revive(
    session: &ZendSession,
    records: &dyn Records,
    owner: &str,
    worker: &mut Worker,
    now: u64,
) -> Result<String, String> {
    let told = resumed_message(worker);
    worker.state = State::Running;
    worker.error = None;
    worker.ended_ms = None;
    // A member that left handed it on (`docs/turn_handover.md` §5): that is
    // explained, and does not count toward the bound on unexplained losses.
    let why = match worker.handed_ms.take() {
        Some(_) => "handed on by a member that left".to_string(),
        None => {
            worker.resumes += 1;
            format!("cut off, try {}", worker.resumes)
        }
    };
    worker.beat_ms = now;
    worker.turn_ms = now;
    worker.entries_after = newest_entry(session.journal(&worker.conv_id).as_ref());
    worker.entries = Vec::new();
    note(records, owner, worker, "Resumed:", why)?;
    Ok(told)
}

/// Start the lost task `name` of `thread` again, in its own conversation
/// (`super::resuming`), under `lease`. `false` when it no longer needs it: read
/// again under the lock a reply takes, it is running, or was answered, or
/// archived.
///
/// A round it stopped in the middle of is finished first, with what it is told
/// following that round's results: a new message would leave those calls
/// unanswered in the conversation.
pub fn resume(
    session: &Arc<ZendSession>,
    owner: &str,
    thread: &str,
    name: &str,
    now: u64,
    lease: Guard,
) -> Result<bool, String> {
    let records = session.shared_substrate().ok_or_else(loading)?;
    let _held = holding();
    // Read under the lock a disposal marks a chat closing under, so a chat
    // being disposed of is never started again.
    if !resuming::chat_open(&records, owner, thread) {
        return Ok(false);
    }
    let Some(found) = threads::worker(&records, owner, thread, name) else {
        return Ok(false);
    };
    let mut worker = found.observed(now);
    if !resuming::due(&worker, now)
        || archiving::being_archived(&worker.conv_id)
        || session.live_turn(&worker.conv_id).is_some()
    {
        return Ok(false);
    }
    let told = revive(session, &records, owner, &mut worker, now)?;
    let first = match session.pending_round(&worker.conv_id) {
        Some(call_turn) => {
            nudges::queue(&records, &worker.conv_id, Sender::Zend, &told, now)
                .map_err(|e| e.to_string())?;
            First::Interrupted { call_turn }
        }
        None => First::Message(format!("{told}{}", closing_of(worker.kind))),
    };
    begin_held(session, &records, owner, worker, first, Some(lease))?;
    Ok(true)
}

/// Finish, as `name` of `thread`'s own, the round at `call_turn` this member's
/// last run stopped in the middle of: the worker is carried on from there, with
/// its beat, its settling and its lease, so no other member takes it for lost
/// and starts it over the same round. `false` when the worker is not one to
/// carry on — it ended, or it was cut off too often — and the
/// round is simply finished. `true` too when another member holds its turns.
pub async fn carry_on(
    session: &Arc<ZendSession>,
    owner: &str,
    thread: &str,
    name: &str,
    call_turn: u32,
) -> Result<bool, String> {
    let records = session.shared_substrate().ok_or_else(loading)?;
    let cut_off = |w: &Worker| {
        let seen = w.clone().observed(now_ms());
        seen.resumes < resuming::MAX_RESUMES
            && (seen.state == State::Running || archiving::lost(&seen))
    };
    let Some(found) = threads::worker(&records, owner, thread, name).filter(cut_off) else {
        return Ok(false);
    };
    let conv_id = found.conv_id;
    let share = tokio::task::spawn_blocking(move || ingest_lease::share_worker(&conv_id))
        .await
        .map_err(|e| e.to_string())?;
    let Share::Mine(lease) = share else {
        return Ok(true);
    };
    let _held = holding();
    let Some(mut worker) = threads::worker(&records, owner, thread, name).filter(cut_off) else {
        return Ok(false);
    };
    let now = now_ms();
    let told = revive(session, &records, owner, &mut worker, now)?;
    nudges::queue(&records, &worker.conv_id, Sender::Zend, &told, now)
        .map_err(|e| e.to_string())?;
    begin_held(
        session,
        &records,
        owner,
        worker,
        First::Interrupted { call_turn },
        Some(lease),
    )?;
    Ok(true)
}
