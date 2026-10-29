//! `repo_task_wait {workers, wait_s?}`: wait on one or more workers, answering as
//! soon as any of them stops running (`docs/claude_zend_gateway.md` §3.3).

use serde_json::{json, Value};

use super::{wait_secs, Answer, Ctx, Wait, LOADING};
use crate::mcp::names::Path;
use crate::mcp::reported::{self, Mark};
use crate::mcp::threads::{State, Worker};
use crate::mcp::workers;

pub const NAME: &str = "repo_task_wait";

/// The line a wait's answer opens its workers still running under.
pub const STILL_RUNNING: &str = "Still running:";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "Wait on running workers, by the paths `repo_task_start` answered (`T:<thread>/<name>`). \
            Answers as soon as any one stops running — done, failed, asking you a question, or stuck — with a \
            line for each still running. So start every task first and wait on them all, handle the one that \
            settled (reply to a question or a stuck worker with `repo_task_reply`), then wait again on the rest. A \
            worker that is done answers with its final response in full. If `wait_s` passes first, each says \
            what it has done so far, with any journal entries it wrote since you last heard: call again, or \
            stop waiting if the entries already answer you. No entry is given twice. The work goes on whether \
            or not anyone waits.",
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
            let checking = w
                .audit
                .as_deref()
                .map(|a| format!("\n  {a}"))
                .unwrap_or_default();
            format!(
                "{} — {}{checking}{entries}",
                w.path(),
                w.doing(workers::now_ms())
            )
        }
        State::Question => format!(
            "{} — asks Claude (reply with `repo_task_reply`; it waits until then):\n{}{entries}",
            w.path(),
            w.question.as_deref().unwrap_or("")
        ),
        State::Stuck if w.stopped => format!(
            "{} — stopped, as you asked; it is live. Redirect it with `repo_task_reply`, or end it \
             with `thread_archive`. Its closing response:\n{}{entries}",
            w.path(),
            w.final_text
        ),
        State::Stuck => format!(
            "{} — stuck: the repeat guard or the round cap closed its tools. Say how to go on \
             with `repo_task_reply`. Its closing response:\n{}{entries}",
            w.path(),
            w.final_text
        ),
        State::Error => format!(
            "{} — error: {}{entries}",
            w.path(),
            w.error.as_deref().unwrap_or("the task failed")
        ),
        State::Done if w.entries.is_empty() => format!(
            "{} — done; no journal entry was kept ({}). Its answer:\n{}{}",
            w.path(),
            w.no_entry.as_deref().unwrap_or("no journal draft ran"),
            w.final_text,
            done_close(w)
        ),
        State::Done => format!(
            "{} — done. Its answer:\n{}{}{entries}",
            w.path(),
            w.final_text,
            done_close(w)
        ),
    }
}

/// What follows a finished task's answer: its citation audit, and that it can
/// be continued while it can.
fn done_close(w: &Worker) -> String {
    let mut out = String::new();
    if let Some(audit) = &w.audit {
        out.push_str("\n\n");
        out.push_str(audit);
    }
    if w.resumable {
        out.push_str(
            "\n\nIt can be continued with `repo_task_reply` — to check a claim, go deeper, or redo a \
             part — until its thread is archived or it sits a day untouched.",
        );
    }
    out
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
        out.push_str(STILL_RUNNING);
        out.push('\n');
        out.push_str(&running.join("\n\n"));
    }
    out
}

/// Wait on `wanted` up to `wait`, telling each step on the call's progress,
/// and answer with [`render`], carrying the entries it hands over as marks to
/// keep once it is delivered, so the next answer gives only newer ones. A wait
/// its client's ceiling cut short says so while anything still runs. What
/// `repo_task_wait`, `repo_task_start` and `repo_task_reply` answer.
pub async fn answer_for(ctx: &Ctx, wanted: &[(String, String)], wait: Wait) -> Answer {
    let owner = ctx.caller.key();
    let read = match workers::wait(
        &ctx.session,
        &owner,
        wanted,
        wait.duration(),
        ctx.progress.as_ref(),
    )
    .await
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
    let mut text = render(&read, &seen);
    let still_running = read
        .iter()
        .any(|(_, w)| w.as_ref().is_some_and(|w| w.state.working()));
    if let Some(note) = wait.note().filter(|_| still_running) {
        text.push_str("\n\n");
        text.push_str(&note);
    }
    Answer::ok(text).marking(marks)
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
    answer_for(ctx, &wanted, wait_secs(ctx, args)).await
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
            effort: Default::default(),
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
            stopped: false,
            resumable: false,
            audit: None,
            checked: false,
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
            "T:review/v — asks Claude (reply with `repo_task_reply`; it waits until then):\n\
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

    /// **A worker sent back to check its citations says so while it runs.**
    #[test]
    fn a_worker_checking_its_citations_says_so_while_it_runs() {
        let mut checking = worker("v", State::Running);
        checking.audit =
            Some("Checking its citations: sent back to open 2 it cited without opening.".into());
        assert_eq!(
            result_of(&checking, None),
            format!(
                "T:review/v — {}\n  Checking its citations: sent back to open 2 it cited without \
                 opening.",
                checking.doing(workers::now_ms())
            )
        );
    }

    /// **A finished task closes with its audit, and says it can be
    /// continued** while it is kept whole.
    #[test]
    fn a_done_task_closes_with_its_audit_and_how_to_continue_it() {
        let mut done = worker("v", State::Done);
        done.entries = vec![entry(4, "Entry 4 — b")];
        done.audit = Some("Citation audit: 1 cited — 1 opened, 0 seen only in search results, 0 never seen in the run.".into());
        done.resumable = true;
        assert_eq!(
            result_of(&done, None),
            "T:review/v — done. Its answer:\nVerdict: ok\n\n\
             Citation audit: 1 cited — 1 opened, 0 seen only in search results, 0 never seen in the run.\n\n\
             It can be continued with `repo_task_reply` — to check a claim, go deeper, or redo a part — \
             until its thread is archived or it sits a day untouched.\n\n\
             Journal entries:\n\nT:review/v#4\nEntry 4 — b"
        );
    }
}
