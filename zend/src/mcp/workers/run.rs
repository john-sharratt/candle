//! The life of a task's turn after it begins: running it under its lease, checking
//! its answer, and settling the worker.
//!
//! The record says the task runs for as long as its turn and its draft do, so no
//! reader takes it for cut off. The lease keeps other members from starting the
//! same turn. An answer that cites what the worker never opened, or that skips
//! steps its brief named, is sent back once, and the second answer is kept with its
//! audit whatever it says. A check round that fails leaves the task done with the
//! answer it had. A failed task is archived with its evidence kept, and a finished
//! task is kept whole until its thread is archived or it sits a day idle.

use super::*;

/// The turn's life after [`begin`], and the worker's settling, under the lease
/// on its turns from start to end: `lease`, or one taken here.
pub(super) async fn run(
    session: Arc<ZendSession>,
    owner: String,
    mut worker: Worker,
    first: First,
    lease: Option<Guard>,
) {
    let Some(records) = session.shared_substrate() else {
        return;
    };
    // The record says the task still runs for as long as it does — through
    // the turn and the draft after it — so no reader takes it for cut off,
    // and carries what the turn has done so far.
    let live = Live::default();
    let (thread, name) = (worker.thread.clone(), worker.name.clone());
    let start_beat = || {
        tokio::spawn(beat(
            Arc::clone(&session),
            records.clone(),
            owner.clone(),
            thread.clone(),
            name.clone(),
            live.clone(),
        ))
    };
    let mut beating = start_beat();
    let mut lease = match lease {
        Some(lease) => Some(lease),
        None => hold_turns(&worker.conv_id).await,
    };
    if lease.is_none() {
        beating.abort();
        let _ = beating.await;
        give_up(&session, &records, &owner, &mut worker);
        session.mcp().bump();
        return;
    }
    let mut next = first;
    // Whether a turn here may move to a markedly faster member: not again,
    // once a handing failed, so a peer that cannot take it is not asked at
    // every round.
    let mut may_hand = true;
    // The answer, audit and settling of a task sent back to open what it
    // cited: what it ends with when the check round does not finish.
    let mut first: Option<(String, Audit, (&'static str, String))> = None;
    let (label, outcome) = loop {
        // A check round does not move: the answer it falls back on is kept
        // here, in `first`.
        let movable = may_hand
            && first.is_none()
            && worker.runner.is_some()
            && worker.runner == worker_hop::own_node();
        // Counted as running through the turn and, when the member leaves
        // under it, the record written after: the leave waits for both.
        let running = session.turn_running();
        let ended = match turn(&session, &worker, next, movable, &live).await {
            Turned::Ended(ended) => {
                drop(running);
                ended
            }
            Turned::Left => {
                beating.abort();
                let _ = beating.await;
                drop(lease.take());
                hand_left(&records, &owner, &worker);
                tracing::info!(
                    conv_id = %worker.conv_id,
                    "mcp: the member leaves — the worker is handed on for another member to resume",
                );
                drop(running);
                return;
            }
            Turned::Handed {
                to,
                call_turn,
                iteration,
            } => {
                // Stopped first, so the record the member taking it writes is
                // the last word on who runs it.
                beating.abort();
                let _ = beating.await;
                // Handed back with the turn's writes, for the member taking it.
                drop(lease.take());
                let handed = hand_on(&session, &owner, &worker, to, call_turn, iteration).await;
                let moved = match handed {
                    Ok(()) => true,
                    Err(why) if taken_within(&records, &owner, &worker, to).await => {
                        tracing::warn!(
                            conv_id = %worker.conv_id,
                            to = %format!("{to:016x}"),
                            "mcp: the handing's answer was lost, but the worker moved: {why}"
                        );
                        true
                    }
                    Err(why) => {
                        tracing::warn!(
                            conv_id = %worker.conv_id,
                            to = %format!("{to:016x}"),
                            "mcp: the worker could not move, so it goes on here: {why}"
                        );
                        false
                    }
                };
                // The member it moved to beats, settles and answers for it
                // now; what Claude sent it, a nudge or a stop, is read there.
                if moved {
                    session.forget(&worker.conv_id);
                    tracing::info!(
                        conv_id = %worker.conv_id,
                        to = %format!("{to:016x}"),
                        call_turn,
                        "mcp: the worker moved to a faster member"
                    );
                    return;
                }
                may_hand = false;
                // Said at once, so the stretch without a beat stays short.
                beat_now(&records, &owner, &worker);
                beating = start_beat();
                lease = hold_turns(&worker.conv_id).await;
                if lease.is_none() {
                    beating.abort();
                    let _ = beating.await;
                    give_up(&session, &records, &owner, &mut worker);
                    session.mcp().bump();
                    return;
                }
                next = First::Round {
                    call_turn,
                    iteration,
                };
                continue;
            }
        };
        live.set_doing(Doing::Journaling);
        // A nudge that came after the turn's last tool round was never read.
        for (_, undelivered) in nudges::take(&records, &worker.conv_id) {
            let line = format!(
                "not delivered — the turn ended without another tool round: {}",
                first_line(&undelivered, PROMPT_QUOTE)
            );
            if let Err(e) = note(&records, &owner, &worker, "Nudge dropped:", line) {
                tracing::warn!(conv_id = %worker.conv_id, "mcp: an undelivered nudge could not be entered: {e}");
            }
        }
        worker.no_entry = None;
        // A chat's answer is ready when its turn ends: no draft stands between
        // Claude and the reply (`docs/zen_chat.md` §2).
        let drafted = match ended {
            Ended::Failed(_) => None,
            _ if worker.kind.is_chat() => None,
            _ => Some(finish_journal(&session, &worker.conv_id).await),
        };
        // A stop that came after the turn's last tool round — while it
        // answered or drafted — found it finishing.
        if stops::take(&records, &worker.conv_id).is_some() {
            let line = "not reached — the turn ended without another tool round".to_string();
            if let Err(e) = note(&records, &owner, &worker, "Stop dropped:", line) {
                tracing::warn!(conv_id = %worker.conv_id, "mcp: an unreached stop could not be entered: {e}");
            }
        }
        // A failed turn still carries what its cadence drafted while it ran.
        let view = session.journal(&worker.conv_id);
        worker.entries = entries_after(view.as_ref(), worker.entries_after);
        if let Some(drafted) = drafted.filter(|_| worker.entries.is_empty()) {
            worker.no_entry = Some(why_no_entry(view.as_ref(), drafted));
        }
        if worker.kind.is_chat() {
            break settle_chat(&mut worker, ended);
        }
        let settled = settle(&mut worker, ended);
        if worker.state != State::Done {
            // A check round that failed, was stopped or asked leaves the task
            // done, with the answer it had: the round said nothing better.
            let Some((answer, audit, (label, outcome))) = first.take() else {
                break settled;
            };
            let round = keep_first_answer(&mut worker, answer, &audit);
            break (label, format!("{outcome}; its answer's check {round}"));
        }
        let audit = match audit_of(&session, &worker).await {
            Ok(audit) => audit,
            // Said, so an answer is never read as citing nothing because its
            // audit could not run.
            Err(e) => {
                worker.audit = Some(format!(
                    "Citation audit: could not be run ({e}); its citations are unchecked."
                ));
                break settled;
            }
        };
        // A task's first answer citing what it never opened, or saying nothing
        // of steps its brief gave it (`super::steps`), goes back once to put
        // that right; the second answer is kept with its audit, whatever it
        // says. A continuation is Claude's to direct, with the audit in hand,
        // so it is not sent back.
        let given = steps::of(&worker.prompt);
        let unsaid = steps::unaccounted(&given, &worker.final_text);
        if (audit.unopened() || !unsaid.is_empty()) && first.is_none() && !worker.checked {
            let mut told = Vec::new();
            let mut why = Vec::new();
            if audit.unopened() {
                let to_correct = audit.miscopied.len() + audit.seen.len() + audit.never_seen.len();
                told.push(audit.correction());
                why.push(format!(
                    "{to_correct} of its answer's citations are not by an address it opened"
                ));
            }
            if !unsaid.is_empty() {
                told.push(steps::correction(&given, &unsaid));
                why.push(format!(
                    "its answer says nothing of {} of the {} steps it was given",
                    unsaid.len(),
                    given.len()
                ));
            }
            let why = why.join(", and ");
            let line = format!("{why}; it is putting that right");
            if let Err(e) = note(&records, &owner, &worker, "Task sent back:", line) {
                tracing::warn!(conv_id = %worker.conv_id, "mcp: a send-back could not be entered: {e}");
            }
            next = First::Message(format!("{SENT_BACK}{}{ASKING}", told.join("\n\n")));
            first = Some((worker.final_text.clone(), audit, settled));
            worker.state = State::Running;
            worker.checked = true;
            // Said on its record while the check runs, so a wait shows why a
            // worker that answered is running again. Only these fields: the
            // beat owns the rest of the record while the turn runs.
            let checking = format!("Checking its answer: sent back because {why}.");
            if let Some(mut stored) =
                threads::worker(&records, &owner, &worker.thread, &worker.name)
            {
                stored.audit = Some(checking);
                stored.checked = true;
                if let Err(e) = threads::put_worker(&records, &owner, &stored) {
                    tracing::warn!(conv_id = %worker.conv_id, "mcp: a send-back could not be kept: {e}");
                }
            }
            continue;
        }
        worker.audit = audit.cites().then(|| audit.render());
        break settled;
    };
    // The beat stops, and is done writing, before the settled record is.
    beating.abort();
    let _ = beating.await;
    // Another member took the worker while this one finished a round it
    // thought its own: the record and the settling are that member's.
    let runner_now =
        threads::worker(&records, &owner, &worker.thread, &worker.name).and_then(|w| w.runner);
    if worker_hop::runs_elsewhere(true, runner_now, worker_hop::own_node()) {
        tracing::warn!(
            conv_id = %worker.conv_id,
            "mcp: another member runs this worker now, so its outcome here is not kept"
        );
        return;
    }
    worker.activity = Some(live.snapshot());
    if !worker.state.live() {
        worker.ended_ms = Some(now_ms());
        // A finished task is kept whole, to be continued with `repo_task_reply`,
        // until its thread is archived or it sits a day idle
        // (`super::archiving`); a failed task is archived now. A chat's
        // conversation is never distilled: disposal tombstones it
        // (`docs/zen_chat.md` §4).
        worker.resumable = worker.state == State::Done;
        if !worker.resumable && !worker.kind.is_chat() {
            match session.archive_worker(&worker.conv_id) {
                Some(Ok(())) | None => {}
                Some(Err(e)) => tracing::warn!(
                    conv_id = %worker.conv_id,
                    "mcp: the worker could not be archived: {e}"
                ),
            }
        }
    }
    let chat = worker.kind.is_chat();
    // Said on the chat before the record says the answer ended, so a reader
    // woken by the record finds the message there.
    if chat {
        talk::answered(&records, &owner, &worker);
    }
    if let Err(e) = threads::put_worker(&records, &owner, &worker) {
        tracing::warn!(conv_id = %worker.conv_id, "mcp: the worker's outcome could not be kept: {e}");
    }
    // Nobody waited on an alone task: its end is kept for its owner's next
    // conversation (`super::outcomes`).
    if worker.driven == Driven::Alone {
        let kept = Outcome {
            at_ms: now_ms(),
            thread: worker.thread.clone(),
            name: worker.name.clone(),
            label: label.to_string(),
            text: outcome.clone(),
        };
        if let Err(e) = outcomes::keep(&records, &owner, &kept) {
            tracing::warn!(conv_id = %worker.conv_id, "mcp: an alone task's outcome could not be kept for its owner: {e}");
        }
    }
    thread_export::spawn_export(
        &records,
        session.mcp().memory(),
        &owner,
        &worker.thread,
        &format!("{} {label}", worker.path()),
        &outcome,
    );
    if let Err(e) = note(&records, &owner, &worker, label, outcome) {
        tracing::warn!(conv_id = %worker.conv_id, "mcp: the outcome could not be entered on the thread: {e}");
    }
    // A chat closed while Zen answered is disposed of once the answer has
    // ended.
    if chat {
        talk::settled(&session, &owner, &worker);
    }
    session.mcp().bump();
}

/// Run one turn of the worker on `message` to its end, as its role, counting
/// what it does into `live`.
pub(super) async fn turn(
    session: &Arc<ZendSession>,
    worker: &Worker,
    first: First,
    may_hand: bool,
    live: &Live,
) -> Turned {
    let mode = access::effective_mode(worker.role, None);
    let mut selection = dial_selection(Some(worker.effort.level()), None, Some(true));
    selection.select(ROLE_SELECTOR, worker.kind.option());
    selection.select(DRIVEN_SELECTOR, worker.driven.option());
    apply_tools_dial(&mut selection, mode);
    let (message, round) = first.into_start();
    let mut stream = session
        .submit_worker(
            message,
            worker.conv_id.clone(),
            mode,
            selection,
            round,
            may_hand,
        )
        .await;
    // The round being streamed, and the last one that ended: a turn runs a
    // round per tool call, and its reply is the last round's alone.
    let mut text = String::new();
    let mut ended = String::new();
    let mut failed: Option<String> = None;
    let mut closed: Option<&'static str> = None;
    let mut handed: Option<(u64, u32, usize)> = None;
    let mut left = false;
    while let Some(item) = stream.next().await {
        if let Ok(item) = &item {
            live.observe(item);
        }
        match item {
            Ok(StreamItem::Token(t)) => text.push_str(&t),
            Ok(StreamItem::TurnEnd { .. }) => ended = std::mem::take(&mut text),
            Ok(StreamItem::Error(e)) => {
                failed.get_or_insert(e);
            }
            Ok(StreamItem::Closing(by)) => closed = Some(by),
            Ok(StreamItem::Handed {
                to,
                call_turn,
                iteration,
            }) => handed = Some((to, call_turn, iteration)),
            Ok(StreamItem::Left) => left = true,
            Ok(_) => {}
            Err(e) => {
                failed.get_or_insert(format!("{e:#}"));
            }
        }
    }
    if left {
        return Turned::Left;
    }
    let last = if text.is_empty() { ended } else { text };
    turned(last, failed, closed, handed)
}
