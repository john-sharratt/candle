//! `zend_wait {workers, wait_s?}`: wait on one or more workers, answering as
//! soon as any of them stops running (`docs/claude_zend_gateway.md` §3.3).

use std::time::Duration;

use serde_json::{json, Value};

use super::{wait_secs, Answer, Ctx, LOADING};
use crate::mcp::names::Path;
use crate::mcp::reported::{self, Mark};
use crate::mcp::threads::{State, Worker};
use crate::mcp::workers;

pub const NAME: &str = "zend_wait";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Wait on running zend workers, by path (`T:<thread>/<name>`, as `zend_task` answered \
            it). Answers as soon as any of them stops running — done with its journal entry, failed, asking you \
            a question, or stuck — and one line for each still running, so start every task first and wait on \
            them all; handle the one that settled (answer a question or a stuck worker with `zend_answer`), \
            then wait again on the rest. A worker that is done answers with its final response in full. If \
            `wait_s` passes first, every one is still running, and each says what it has done so far — its \
            rounds, thinking and tool calls — with any journal entries it has written since you last heard: \
            call again, or stop waiting if the entries already answer you. Every answer gives only entries \
            you have not been given before. The work runs on zend whether or not anyone waits.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "workers": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "The workers' paths."},
                "wait_s": {"type": "integer", "minimum": 0, "description": "How long to wait before answering with every one still running (45 by default)."}
            },
            "required": ["workers"],
            "additionalProperties": false
        }
    })
}

/// The worker's journal entries after `seen` — the newest one already handed
/// to the caller — each under its address.
fn new_entries(w: &Worker, seen: Option<u64>) -> String {
    let mut out = String::new();
    for e in w.entries.iter().filter(|e| seen.is_none_or(|s| e.id > s)) {
        out.push_str(&format!("\n\n{}#{}\n{}", w.path(), e.id, e.text));
    }
    if out.is_empty() {
        out
    } else {
        format!("\n\nJournal entries:{out}")
    }
}

/// One worker's result, as a block of text, giving only the journal entries
/// after `seen`: what a wait answers for it, and what `sub_run` answers for a
/// part.
pub fn result_of(w: &Worker, seen: Option<u64>) -> String {
    let entries = new_entries(w, seen);
    match w.state {
        State::Running => {
            let activity = w.activity.clone().unwrap_or_default();
            format!(
                "{} — {}{entries}",
                w.path(),
                activity.sentence(workers::now_ms().saturating_sub(w.turn_ms))
            )
        }
        State::Question => format!(
            "{} — asks Claude (answer with `zend_answer`; it waits until then):\n{}{entries}",
            w.path(),
            w.question.as_deref().unwrap_or("")
        ),
        State::Stuck => format!(
            "{} — stuck: the repeat guard or the round cap closed its tools. Say how to go on \
             with `zend_answer`. Its closing response:\n{}{entries}",
            w.path(),
            w.final_text
        ),
        State::Error => format!(
            "{} — error: {}{entries}",
            w.path(),
            w.error.as_deref().unwrap_or("the task failed")
        ),
        State::Done if w.entries.is_empty() => format!(
            "{} — done; no journal entry was kept ({}). Its answer:\n{}",
            w.path(),
            w.no_entry.as_deref().unwrap_or("no journal draft ran"),
            w.final_text
        ),
        State::Done => format!(
            "{} — done. Its answer:\n{}{entries}",
            w.path(),
            w.final_text
        ),
    }
}

/// What a wait answers: every worker that stopped running, in full, then each
/// still running — what its turn has done — with only the journal entries
/// after the newest one `seen` says the caller already has.
pub fn render(read: &[(String, Option<Worker>)], seen: &dyn Fn(&Worker) -> Option<u64>) -> String {
    let mut ended = Vec::new();
    let mut running = Vec::new();
    for (path, w) in read {
        match w {
            None => ended.push(format!("{path} — no such worker")),
            Some(w) if w.state.working() => running.push(result_of(w, seen(w))),
            Some(w) => ended.push(result_of(w, seen(w))),
        }
    }
    let mut out = ended.join("\n\n");
    if !running.is_empty() {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str("Still running:\n");
        out.push_str(&running.join("\n\n"));
    }
    out
}

/// Wait on `wanted` up to `wait`, telling each step on the call's progress,
/// and answer with [`render`], carrying the entries it hands over as marks to
/// keep once it is delivered, so the next answer gives only newer ones. What
/// `zend_wait`, `zend_task` and `zend_answer` answer.
pub async fn answer_for(ctx: &Ctx, wanted: &[(String, String)], wait: Duration) -> Answer {
    let owner = ctx.caller.key();
    let read = match workers::wait(&ctx.session, &owner, wanted, wait, ctx.progress.as_ref()).await
    {
        Ok(read) => read,
        Err(e) => return Answer::error(e),
    };
    let Some(records) = ctx.session.shared_substrate() else {
        return Answer::error(LOADING);
    };
    let seen = |w: &Worker| reported::last(&records, &owner, &w.thread, &w.name);
    let marks = read
        .iter()
        .filter_map(|(_, w)| {
            let w = w.as_ref()?;
            Some(Mark {
                thread: w.thread.clone(),
                name: w.name.clone(),
                entry: w.entries.iter().map(|e| e.id).max()?,
            })
        })
        .collect();
    Answer::ok(render(&read, &seen)).marking(marks)
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    let Some(list) = args.get("workers").and_then(Value::as_array) else {
        return Answer::error("`workers` is missing: a list of worker paths");
    };
    let mut wanted = Vec::new();
    for item in list {
        let Some(text) = item.as_str() else {
            return Answer::error("each of `workers` is a path string");
        };
        match Path::parse(text) {
            Ok(Path::Worker { thread, worker }) => wanted.push((thread, worker)),
            Ok(_) => {
                return Answer::error(format!(
                    "`{text}` is not a worker; a worker is `T:<thread>/<name>`"
                ))
            }
            Err(e) => return Answer::error(e),
        }
    }
    if wanted.is_empty() {
        return Answer::error("`workers` is empty");
    }
    answer_for(ctx, &wanted, Duration::from_secs(wait_secs(ctx, args))).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::activity::{Activity, Doing};
    use crate::mcp::threads::Entry;
    use web::auth::Role;

    fn worker(name: &str, state: State) -> Worker {
        Worker {
            thread: "review".into(),
            name: name.into(),
            conv_id: "c".into(),
            prompt: "p".into(),
            state,
            role: Role::User,
            priority: Default::default(),
            started_ms: workers::now_ms(),
            turn_ms: workers::now_ms(),
            beat_ms: 0,
            ended_ms: None,
            entries_after: None,
            entries: Vec::new(),
            final_text: "Verdict: ok".into(),
            error: None,
            question: None,
            activity: None,
            no_entry: None,
        }
    }

    fn entry(id: u64, text: &str) -> Entry {
        Entry {
            id,
            text: text.into(),
        }
    }

    #[test]
    fn a_question_and_a_stuck_worker_say_what_to_answer() {
        let mut asks = worker("v", State::Question);
        asks.question = Some("QUESTION: Inclusive?".into());
        assert_eq!(
            result_of(&asks, None),
            "T:review/v — asks Claude (answer with `zend_answer`; it waits until then):\n\
             QUESTION: Inclusive?"
        );
        let stuck = worker("v", State::Stuck);
        assert!(result_of(&stuck, None).contains("closed its tools"));
        assert!(result_of(&stuck, None).ends_with("Verdict: ok"));
    }

    /// **A running worker says what it has done**, and the entries it wrote
    /// that the caller has not been given; the ended come first, in full.
    #[test]
    fn the_ended_come_first_and_the_running_say_what_they_have_done() {
        let mut done = worker("verify", State::Done);
        done.entries = vec![entry(4, "Entry 4 — verified")];
        let mut check = worker("check", State::Running);
        check.activity = Some(Activity {
            rounds: 2,
            thinking: 1_934,
            tools: [("web_fetch".to_string(), 3)].into_iter().collect(),
            now: Doing::Thinking,
            ..Activity::default()
        });
        check.entries = vec![
            entry(1, "Entry 1 — read the plan"),
            entry(2, "Entry 2 — fetched"),
        ];
        let read = vec![
            ("T:review/check".to_string(), Some(check)),
            ("T:review/verify".to_string(), Some(done)),
            ("T:review/gone".to_string(), None),
        ];
        let seen = |w: &Worker| (w.name == "check").then_some(1);
        assert_eq!(
            render(&read, &seen),
            "T:review/verify — done. Its answer:\nVerdict: ok\n\n\
             Journal entries:\n\nT:review/verify#4\nEntry 4 — verified\n\n\
             T:review/gone — no such worker\n\n\
             Still running:\n\
             T:review/check — running 0s, 2 rounds: 1,934 thinking tokens, 3 tool calls \
             (web_fetch 3); now thinking\n\n\
             Journal entries:\n\nT:review/check#2\nEntry 2 — fetched"
        );
    }

    /// **Done gives the final response whole**, and only the entries the
    /// caller has not been given; with none kept, it says so.
    #[test]
    fn a_done_task_gives_its_answer_and_only_unseen_entries() {
        let mut done = worker("v", State::Done);
        done.entries = vec![entry(3, "Entry 3 — a"), entry(4, "Entry 4 — b")];
        assert_eq!(
            result_of(&done, Some(4)),
            "T:review/v — done. Its answer:\nVerdict: ok"
        );
        assert_eq!(
            result_of(&done, Some(3)),
            "T:review/v — done. Its answer:\nVerdict: ok\n\n\
             Journal entries:\n\nT:review/v#4\nEntry 4 — b"
        );
        let mut bare = worker("v", State::Done);
        bare.no_entry =
            Some("its journal draft failed: the entry was refused: Claim 1 cites no turn.".into());
        assert_eq!(
            result_of(&bare, None),
            "T:review/v — done; no journal entry was kept (its journal draft failed: the entry was \
             refused: Claim 1 cites no turn.). Its answer:\nVerdict: ok"
        );
        let mut failed = worker("v", State::Error);
        failed.error = Some("boom".into());
        assert_eq!(result_of(&failed, None), "T:review/v — error: boom");
    }
}
