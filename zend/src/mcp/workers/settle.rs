//! Settles a worker on how its turn ended, and checks its answer's citations.
//!
//! A worker's final response is audited against every turn of its conversation.
//! When a check round did not finish, the answer goes back to the one the worker
//! gave first, so a failed check cannot leave the worker with nothing. A chat's
//! side settles as done with its reply and never as a question, because Zen asks
//! Claude within the chat itself, and its thread keeps no line about it.

use super::*;

/// The citation audit of `worker`'s final response, over every turn of its
/// conversation (`super::audit`): decoding them reads the substrate, so it
/// runs off the runtime's threads.
pub(super) async fn audit_of(session: &Arc<ZendSession>, worker: &Worker) -> Result<Audit, String> {
    let own = worker.conv_id.clone();
    let session = Arc::clone(session);
    let answer = worker.final_text.clone();
    let audited = tokio::task::spawn_blocking(move || {
        let detail: Option<TimelineDetail> = session.conversation_detail(&own);
        let turns: Vec<Turn<'_>> = detail
            .iter()
            .flat_map(|d| &d.turns)
            .filter(|t| t.kind == "normal")
            .map(|t| t.user.as_str())
            .collect();
        Audit::of(&answer, &turns)
    })
    .await;
    audited.map_err(|e| {
        tracing::warn!(conv_id = %worker.conv_id, "mcp: the citation audit failed: {e}");
        e.to_string()
    })
}

/// Settle `worker` back on the `answer` it gave before it was sent to open
/// what it cited, when the check round ended any way but done: done, with
/// that answer, its `audit`, and what became of the round — which it answers,
/// for the thread's line.
pub(super) fn keep_first_answer(worker: &mut Worker, answer: String, audit: &Audit) -> String {
    let round = match worker.state {
        State::Error => format!(
            "failed: {}",
            first_line(worker.error.as_deref().unwrap_or("the turn failed"), 200)
        ),
        State::Question => format!(
            "ended asking: {}",
            first_line(worker.question.as_deref().unwrap_or(""), 200)
        ),
        _ if worker.stopped => "was stopped by Claude".to_string(),
        _ => "had its tools closed before it finished".to_string(),
    };
    worker.audit = Some(format!(
        "{}\nIt was sent back to open these, and the check round {round}; the answer above is \
         the one it gave before. Continue it with `repo_task_reply` to finish the check.",
        audit.render()
    ));
    worker.final_text = answer;
    worker.state = State::Done;
    worker.error = None;
    worker.question = None;
    worker.stopped = false;
    round
}

/// Settle `worker` on how its turn `ended`: its new state, and the thread's
/// line about it.
pub(super) fn settle(worker: &mut Worker, ended: Ended) -> (&'static str, String) {
    let cite = |w: &Worker| match w.entries.last() {
        Some(e) => format!(" ({}#{})", w.path(), e.id),
        None => String::new(),
    };
    worker.stopped = ended.stopped();
    match ended {
        Ended::Failed(error) => {
            let line = format!("error: {}", first_line(&error, 200));
            worker.error = Some(error);
            worker.state = State::Error;
            ("Task ended:", line)
        }
        Ended::Closed { text, .. } if worker.stopped => {
            worker.final_text = final_reply(&text);
            worker.state = State::Stuck;
            let line = format!(
                "stopped by Claude; it ended: {}{}",
                first_line(&worker.final_text, 200),
                cite(worker)
            );
            ("Task stopped:", line)
        }
        Ended::Closed { text, .. } => {
            worker.final_text = final_reply(&text);
            worker.state = State::Stuck;
            let line = format!(
                "stuck: its tools were closed; it ended: {}{}",
                first_line(&worker.final_text, 200),
                cite(worker)
            );
            ("Task stuck:", line)
        }
        Ended::Replied(text) => {
            worker.final_text = final_reply(&text);
            match question_of(&worker.final_text) {
                Some(q) => {
                    let line = format!("{}{}", first_line(&q, 200), cite(worker));
                    worker.question = Some(q);
                    worker.state = State::Question;
                    ("Task asks:", line)
                }
                None => {
                    worker.state = State::Done;
                    let line = match worker.entries.last() {
                        Some(e) => format!("{}{}", summary_line(&e.text, 200), cite(worker)),
                        None => format!(
                            "no entry was kept; it ended: {}",
                            first_line(&worker.final_text, 200)
                        ),
                    };
                    ("Task ended:", line)
                }
            }
        }
    }
}

/// Settle Zen's side of a chat on how its turn `ended`: `done` with its reply —
/// a closed loop's text still its reply, and a question block no question,
/// since Zen asks Claude in the chat itself — or `error`. The line is for no
/// journal: a chat's thread keeps none ([`note`]).
pub(super) fn settle_chat(worker: &mut Worker, ended: Ended) -> (&'static str, String) {
    worker.stopped = ended.stopped();
    match ended {
        Ended::Failed(error) => {
            let line = format!("error: {}", first_line(&error, 200));
            worker.error = Some(error);
            worker.state = State::Error;
            ("Chat answer:", line)
        }
        Ended::Closed { text, .. } | Ended::Replied(text) => {
            worker.final_text = final_reply(&text);
            worker.state = State::Done;
            ("Chat answer:", "done".into())
        }
    }
}
