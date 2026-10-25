//! `zend_wait {workers, wait_s?}`: wait on one or more workers, answering as
//! soon as any of them stops running (`docs/claude_zend_gateway.md` §3.3).

use std::time::Duration;

use serde_json::{json, Value};

use super::{wait_secs, Answer, Ctx};
use crate::mcp::names::Path;
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
            then wait again on the rest. If `wait_s` passes first, every one is still running: call again. The \
            work runs on zend whether or not anyone waits.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "workers": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "The workers' paths."},
                "wait_s": {"type": "integer", "minimum": 0, "description": "How long to wait before answering with every one still running (50 by default)."}
            },
            "required": ["workers"],
            "additionalProperties": false
        }
    })
}

/// One worker's result, as a block of text.
fn result_of(w: &Worker) -> String {
    match w.state {
        State::Running => format!("{} — running", w.path()),
        State::Question => format!(
            "{} — asks Claude (answer with `zend_answer`; it waits until then):\n{}",
            w.path(),
            w.question.as_deref().unwrap_or("")
        ),
        State::Stuck => format!(
            "{} — stuck: the repeat guard or the round cap closed its tools. Say how to go on \
             with `zend_answer`. Its closing response:\n{}",
            w.path(),
            w.final_text
        ),
        State::Error => format!(
            "{} — error: {}",
            w.path(),
            w.error.as_deref().unwrap_or("the task failed")
        ),
        State::Done if w.entries.is_empty() => format!(
            "{} — done; no journal entry was kept. Its final response ended:\n{}",
            w.path(),
            w.final_text
        ),
        State::Done => {
            let mut out = format!("{} — done. Its journal:", w.path());
            for e in &w.entries {
                out.push_str(&format!("\n\n{}#{}\n{}", w.path(), e.id, e.text));
            }
            out
        }
    }
}

/// What a wait answers: every worker that stopped running, in full, then one
/// line per worker still running.
pub fn render(read: &[(String, Option<Worker>)]) -> String {
    let mut ended = Vec::new();
    let mut running = Vec::new();
    for (path, w) in read {
        match w {
            None => ended.push(format!("{path} — no such worker")),
            Some(w) if w.state.working() => running.push(format!(
                "{} — running since {} s ago",
                w.path(),
                workers::now_ms().saturating_sub(w.started_ms) / 1000
            )),
            Some(w) => ended.push(result_of(w)),
        }
    }
    let mut out = ended.join("\n\n");
    if !running.is_empty() {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str("Still running:\n");
        out.push_str(&running.join("\n"));
    }
    out
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
            Ok(Path::Thread(_)) => {
                return Answer::error(format!(
                    "`{text}` is a thread; a worker is `T:<thread>/<name>`"
                ))
            }
            Err(e) => return Answer::error(e),
        }
    }
    if wanted.is_empty() {
        return Answer::error("`workers` is empty");
    }
    let wait = Duration::from_secs(wait_secs(ctx, args));
    match workers::wait(&ctx.session, &ctx.caller.key(), &wanted, wait).await {
        Ok(read) => Answer::ok(render(&read)),
        Err(e) => Answer::error(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::threads::Entry;

    fn worker(name: &str, state: State) -> Worker {
        Worker {
            thread: "review".into(),
            name: name.into(),
            conv_id: "c".into(),
            prompt: "p".into(),
            state,
            started_ms: workers::now_ms(),
            beat_ms: 0,
            ended_ms: None,
            entries_after: None,
            entries: Vec::new(),
            final_text: "Verdict: ok".into(),
            error: None,
            question: None,
        }
    }

    #[test]
    fn a_question_and_a_stuck_worker_say_what_to_answer() {
        let mut asks = worker("v", State::Question);
        asks.question = Some("QUESTION: Inclusive?".into());
        assert_eq!(
            result_of(&asks),
            "T:review/v — asks Claude (answer with `zend_answer`; it waits until then):\n\
             QUESTION: Inclusive?"
        );
        let stuck = worker("v", State::Stuck);
        assert!(result_of(&stuck).contains("closed its tools"));
        assert!(result_of(&stuck).ends_with("Verdict: ok"));
    }

    #[test]
    fn the_ended_come_first_in_full_and_the_running_as_lines() {
        let mut done = worker("verify", State::Done);
        done.entries = vec![Entry {
            id: 4,
            text: "Entry 4 — verified".into(),
        }];
        let read = vec![
            (
                "T:review/check".to_string(),
                Some(worker("check", State::Running)),
            ),
            ("T:review/verify".to_string(), Some(done)),
            ("T:review/gone".to_string(), None),
        ];
        assert_eq!(
            render(&read),
            "T:review/verify — done. Its journal:\n\nT:review/verify#4\nEntry 4 — verified\n\n\
             T:review/gone — no such worker\n\n\
             Still running:\nT:review/check — running since 0 s ago"
        );
    }

    #[test]
    fn a_done_task_with_no_entry_shows_its_reply_and_an_error_its_cause() {
        let done = worker("v", State::Done);
        assert!(result_of(&done).contains("no journal entry was kept"));
        assert!(result_of(&done).ends_with("Verdict: ok"));
        let mut failed = worker("v", State::Error);
        failed.error = Some("boom".into());
        assert_eq!(result_of(&failed), "T:review/v — error: boom");
    }
}
