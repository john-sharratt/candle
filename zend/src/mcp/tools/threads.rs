//! `thread_list {include_archived?}`: the caller's threads, newest first, and
//! how each stands (`docs/claude_zend_gateway.md` §3.2, §6.2).

use serde_json::{json, Value};

use super::{Answer, Ctx};
use crate::mcp::threads::{self, State, Thread, Worker};
use crate::mcp::workers::{first_line, now_ms};

pub const NAME: &str = "thread_list";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "List your threads, newest first: each one's path, how many tasks it has had, each \
            worker still running with what it has done so far, and any waiting on you — a question to answer, \
            or a stuck worker — with its path. Use it to find the thread of a feature you worked on before, \
            and to see what is waiting on you when no call was open to return it. A thread left a day with \
            nothing new is archived on its own, as is any worker left a day.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "include_archived": {"type": "boolean", "description": "Also list archived threads (false by default)."}
            },
            "additionalProperties": false
        }
    })
}

/// One thread's line, then a line for each worker running — what its turn
/// has done, at `now` — and each waiting on Claude.
pub fn render_thread(t: &Thread, workers: &[Worker], last: Option<&str>, now: u64) -> String {
    let running = workers.iter().filter(|w| w.state.working()).count();
    let mut out = format!(
        "T:{}{} — {} task{}",
        t.name,
        if t.archived { " (archived)" } else { "" },
        workers.len(),
        if workers.len() == 1 { "" } else { "s" }
    );
    if running > 0 {
        out.push_str(&format!(", {running} running"));
    }
    if let Some(last) = last {
        out.push_str(&format!("; last: {last}"));
    }
    for w in workers {
        match w.state {
            State::Running => out.push_str(&format!("\n  {} — {}", w.path(), w.doing(now))),
            State::Question => out.push_str(&format!(
                "\n  {} asks: {}",
                w.path(),
                first_line(w.question.as_deref().unwrap_or(""), 200)
            )),
            State::Stuck if w.stopped => {
                out.push_str(&format!("\n  {} is stopped, as you asked", w.path()))
            }
            State::Stuck => out.push_str(&format!("\n  {} is stuck", w.path())),
            _ => {}
        }
    }
    out
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let include_archived = args
        .get("include_archived")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let Some(records) = ctx.session.shared_substrate() else {
        return Answer::error("the model is still loading; try again shortly");
    };
    let owner = ctx.caller.key();
    let now = now_ms();
    let lines: Vec<String> = threads::threads(&records, &owner)
        .into_iter()
        .filter(|t| include_archived || !t.archived)
        .map(|t| {
            // A thread's tasks are Claude's; the parts its workers started
            // are theirs, and wait on nobody.
            let workers: Vec<Worker> = threads::workers(&records, &owner, &t.name)
                .into_iter()
                .filter(|w| !w.is_part())
                .map(|w| w.observed(now))
                .collect();
            let last = threads::last_note(&records, &owner, &t.name)
                .map(|n| first_line(&format!("{} {}", n.label, n.text), 160));
            render_thread(&t, &workers, last.as_deref(), now)
        })
        .collect();
    if lines.is_empty() {
        return Answer::ok("You have no threads yet: `thread_attach` starts one.");
    }
    Answer::ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
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
            started_ms: 0,
            turn_ms: 0,
            beat_ms: 0,
            ended_ms: None,
            entries_after: None,
            entries: Vec::new(),
            final_text: String::new(),
            error: None,
            question: Some("QUESTION: Inclusive?".into()),
            activity: None,
            no_entry: None,
            stopped: false,
            resumable: false,
            resumes: 0,
            audit: None,
            checked: false,
            runner: None,
        }
    }

    #[test]
    fn a_thread_names_what_runs_and_what_waits_on_claude() {
        let t = Thread {
            name: "review".into(),
            created_ms: 0,
            archived: false,
            attached_ms: 0,
        };
        let ws = [
            worker("a", State::Running),
            worker("b", State::Question),
            worker("c", State::Stuck),
            worker("d", State::Done),
        ];
        assert_eq!(
            render_thread(&t, &ws, Some("Task ended: done"), 250_000),
            "T:review — 4 tasks, 1 running; last: Task ended: done\n  \
             T:review/a — running 4m10s, 0 rounds: 0 thinking tokens, no tool calls yet; now \
             starting\n  \
             T:review/b asks: QUESTION: Inclusive?\n  T:review/c is stuck"
        );
        let archived = Thread {
            archived: true,
            ..t
        };
        assert_eq!(
            render_thread(&archived, &[], None, 0),
            "T:review (archived) — 0 tasks"
        );
    }
}
