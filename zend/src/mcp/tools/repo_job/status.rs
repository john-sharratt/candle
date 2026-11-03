//! `repo_job_status {job, wait_s?}`: where a job is, waiting up to `wait_s`
//! for it to end (`docs/build_deploy_tools.md` §2, §5). One call holds while
//! the job runs, saying each step on the call's progress, so a caller waits in
//! one call rather than polling; an ended job's answer carries its summary —
//! test results, errors, failures by line — so the outcome needs no second
//! call.

use std::time::{Duration, Instant};

use agent::client::{Agents, At};
use agent::wire::JobView;
use cluster::log_read::{Line, LogAsk, LogRead, Summary};
use serde_json::{json, Value};

use super::{describe, host_of, lost, phase, Handle, NO_AGENTS};
use crate::mcp::ops;
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::tools::{string, Answer, Ctx};
use crate::mcp::workers::now_ms;

pub const NAME: &str = "repo_job_status";

/// The seconds a call waits when it names no `wait_s`.
const DEFAULT_WAIT_S: u64 = 60;

/// The longest one wait on the agent is held: under the agent's own cap, so
/// each wakes in time to say how the job stands.
const STEP: Duration = Duration::from_secs(25);

/// The most of each kind of line an ended job's answer quotes.
const QUOTED: usize = 6;

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "Say where a build-agent job is — the machine it runs on, its command, its phase \
            (fetching origin, waiting for the checkout behind named jobs, running, staging what it built, \
            ended), how long it has run, how many lines it has printed and the last of them — and wait up \
            to `wait_s` seconds (60 by default; 0 answers at once) for it to end. Waiting is one call: it \
            holds until the job ends or the wait runs out, so call it once with a long `wait_s` rather than \
            again and again. Once the job has ended the answer says how — exit code, timed out, cancelled, \
            failed before it ran — what it staged, and its summary: cargo's test results and the first \
            errors and failures with their line numbers, which `repo_job_log` with `mode: lines` reads \
            around. `job` is the handle `repo_job_start`, `service_deploy` or `repo_job_list` answered, \
            `<node>:<id>`. A deploy's build is a job: its handle reads here too. Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job": {"type": "string", "minLength": 18, "maxLength": 64, "description": "The job's handle, `<node>:<id>`, e.g. `76c3a0f4d39bd9db:XeV_I_QPzpg`."},
                "wait_s": {"type": "integer", "minimum": 0, "maximum": 3600, "description": "Seconds to wait for it to end; 60 by default, 0 to answer at once. The client's own limit caps it."}
            },
            "required": ["job"],
            "additionalProperties": false
        }
    })
}

/// Up to [`QUOTED`] of `lines`, numbered, under `head`, and how many of the
/// `count` there were are left out.
fn quoted(out: &mut String, head: &str, lines: &[Line], count: u64) {
    if lines.is_empty() {
        return;
    }
    let more = count.saturating_sub(lines.len().min(QUOTED) as u64);
    out.push_str(&format!("\n{head}:"));
    for l in lines.iter().take(QUOTED) {
        out.push_str(&format!("\n  {}: {}", l.n, l.text));
    }
    if more > 0 {
        out.push_str(&format!("\n  …and {more} more"));
    }
}

/// What an ended job's summary says, and where to read on from.
pub fn outcome(h: &Handle, s: &Summary) -> String {
    let mut out = String::new();
    quoted(&mut out, "test results", &s.results, s.results.len() as u64);
    quoted(&mut out, "failures", &s.failures, s.failure_count);
    quoted(&mut out, "errors", &s.errors, s.error_count);
    if s.warning_count > 0 {
        out.push_str(&format!("\n{} warnings", s.warning_count));
    }
    let first = s
        .errors
        .first()
        .into_iter()
        .chain(s.failures.first())
        .map(|l| l.n)
        .min();
    match first {
        Some(n) => out.push_str(&format!(
            "\nRead around it: `repo_job_log {{job: \"{h}\", mode: \"lines\", from: {}, count: 60}}`.",
            n.saturating_sub(10).max(1)
        )),
        None if s.results.is_empty() => {
            if let Some(l) = s.last.last() {
                out.push_str(&format!("\nlast line {}: {}", l.n, l.text));
            }
        }
        None => {}
    }
    out
}

/// The last line of job `h`'s log, numbered, if it has one.
async fn last_line(agents: &Agents, at: At, h: &Handle) -> Option<String> {
    let read: LogRead = agents
        .read_log(at, &h.id, &LogAsk::Tail { lines: 1 })
        .await
        .ok()?;
    read.lines
        .last()
        .map(|l| format!("line {}: {}", l.n, l.text))
}

pub async fn run(call: &OpsCall, args: &Value) -> Answer {
    let h = match string(args, "job").and_then(Handle::parse) {
        Ok(h) => h,
        Err(e) => return Answer::error(e),
    };
    let Some(agents) = call.session.agents() else {
        return Answer::error(NO_AGENTS);
    };
    let (secs, asked) = call.wait_of(args, DEFAULT_WAIT_S);
    let at = h.at(&agents);
    let host = host_of(&agents, h.node).await;
    let until = Instant::now() + Duration::from_secs(secs);
    let mut view: JobView = match agents.view(at, &h.id).await {
        Ok(v) => v,
        Err(e) => return Answer::error(lost(&h, &e)),
    };
    while !view.status.is_finished() {
        let left = until.saturating_duration_since(Instant::now());
        if left.is_zero() {
            break;
        }
        view = match agents.wait(at, &h.id, left.min(STEP)).await {
            Ok(v) => v,
            Err(e) => return Answer::error(lost(&h, &e)),
        };
        if !view.status.is_finished() {
            let last = last_line(&agents, at, &h).await.unwrap_or_default();
            call.say(format!(
                "{} on {host}: {}, {} lines; {last}",
                h,
                phase(&view.phase, h.node),
                view.lines
            ));
        }
    }
    let mut out = describe(&h, &host, &view, now_ms());
    if view.status.is_finished() {
        if let Ok(read) = agents.read_log(at, &h.id, &LogAsk::Summary).await {
            if let Some(s) = &read.summary {
                out.push_str(&outcome(&h, s));
            }
        }
    } else {
        if let Some(last) = last_line(&agents, at, &h).await {
            out.push_str(&format!("\nlast {last}"));
        }
        let cut = asked.map_or(String::new(), |a| {
            format!(" (you asked for {a} s; this client allows {secs})")
        });
        out.push_str(&format!(
            "\nStill going after {secs} s{cut}: call `repo_job_status {{job: \"{h}\", wait_s: {}}}` \
             again to keep waiting.",
            call.ceiling.min(600)
        ));
    }
    Answer::ok(out)
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    run(&OpsCall::of(ctx), args).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(n: u64, text: &str) -> Line {
        Line {
            n,
            text: text.into(),
        }
    }

    /// **An ended job's summary quotes its results, failures and errors by
    /// line, and says where to read around the first of them.**
    #[test]
    fn an_ended_job_says_how_and_where_to_read_on() {
        let s = Summary {
            results: vec![line(90, "test result: FAILED. 9 passed; 1 failed")],
            failures: vec![
                line(70, "test a::b ... FAILED"),
                line(80, "panicked at a.rs:3"),
            ],
            errors: Vec::new(),
            failure_count: 2,
            error_count: 0,
            warning_count: 3,
            last: vec![line(91, "error: test failed")],
        };
        let h = Handle::new(0xab, "Xe");
        assert_eq!(
            outcome(&h, &s),
            "\ntest results:\n  90: test result: FAILED. 9 passed; 1 failed\nfailures:\n  70: test \
             a::b ... FAILED\n  80: panicked at a.rs:3\n3 warnings\nRead around it: \
             `repo_job_log {job: \"00000000000000ab:Xe\", mode: \"lines\", from: 60, count: 60}`."
        );
    }

    /// A list past what is quoted says how many more there were.
    #[test]
    fn a_long_list_says_how_many_more() {
        let errors: Vec<Line> = (1..=9)
            .map(|n| line(n, "error[E0308]: mismatched"))
            .collect();
        let s = Summary {
            errors,
            error_count: 40,
            ..Summary::default()
        };
        let text = outcome(&Handle::new(1, "x"), &s);
        assert!(
            text.contains("\n  6: error[E0308]: mismatched\n  …and 34 more"),
            "{text}"
        );
        assert!(!text.contains("\n  7:"), "{text}");
        assert!(text.contains("from: 1, count: 60"), "{text}");
    }

    /// A job that printed no results or failures shows its last line.
    #[test]
    fn a_quiet_job_shows_its_last_line() {
        let s = Summary {
            last: vec![line(4, "Finished `release` profile")],
            ..Summary::default()
        };
        assert_eq!(
            outcome(&Handle::new(1, "x"), &s),
            "\nlast line 4: Finished `release` profile"
        );
    }
}
