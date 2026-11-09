//! `job_read {job, page, wait_s?}`: one page of a job's output
//! (`docs/jobs.md` §3). A page the job has not finished writing waits until it
//! fills, the job ends, or `wait_s` passes — `tail -f` in bounded chunks — and
//! a page past the end answers the last page.

use std::time::Duration;

use serde_json::{json, Value};
use tokio::time::Instant;

use super::source::{read, standing};
use super::status::{allowed, job_schema};
use super::wait::pause;
use super::{page_ask, pages, JobHandle, LOOK_EVERY, PAGE_LINES};
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::tools::{string, Answer, Ctx};

pub const NAME: &str = "job_read";

/// The seconds a call waits for its page to fill when it names no `wait_s`.
const DEFAULT_WAIT_S: u64 = 30;

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "Read one page of a job's output — 200 lines, numbered across the whole output, so line \
            numbers never move. A build's output is its log; a deploy's, its build's log then its trial's; a \
            task's, its prompt, journal entries, answer and turn report; a maintenance job's, its steps. A \
            page the job is still writing waits until it fills, the job ends, or `wait_s` passes (30 by \
            default), then answers what is there: follow a running job by reading the next page as each one \
            fills. A page past the end answers the last page. `job_status` says how many pages there are \
            and which one a failure is on.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job": job_schema(),
                "page": {"type": "integer", "minimum": 1, "description": "The page, from 1; one past the end reads the last."},
                "wait_s": {"type": "integer", "minimum": 0, "maximum": 3600, "description": "Seconds to wait for the page to fill; 30 by default, 0 to answer at once."}
            },
            "required": ["job", "page"],
            "additionalProperties": false
        }
    })
}

pub async fn run(call: &OpsCall, args: &Value) -> Answer {
    let job = match string(args, "job").and_then(JobHandle::parse) {
        Ok(j) => j,
        Err(e) => return Answer::error(e),
    };
    if let Err(e) = allowed(call, &job) {
        return Answer::error(e);
    }
    let asked = args.get("page").and_then(Value::as_u64).unwrap_or(1).max(1);
    let (secs, _) = call.wait_of(args, DEFAULT_WAIT_S);
    let until = Instant::now() + Duration::from_secs(secs);
    // A task's page fills as it changes, which wakes the wait; any other job
    // is looked at on the cadence.
    let mut wake = matches!(job, JobHandle::Task { .. }).then(|| call.session.mcp().watch());
    loop {
        let s = match standing(call, &job).await {
            Ok(s) => s,
            Err(e) => return Answer::error(e),
        };
        let (page, ask) = page_ask(asked, s.total);
        let r = match read(call, &job, &ask).await {
            Ok(r) => r,
            Err(e) => return Answer::error(e),
        };
        let full = r.lines.len() as u64 >= PAGE_LINES;
        if full || s.ended || Instant::now() >= until {
            let n = pages(r.total.max(s.total));
            let mut out = format!(
                "{job}: page {page} of {n}{}{}",
                if page < asked {
                    format!(" (you asked for page {asked}; this is the last)")
                } else {
                    String::new()
                },
                if s.ended { "" } else { ", still being written" }
            );
            for l in &r.lines {
                out.push_str(&format!("\n{}: {}", l.n, l.text));
            }
            if page < n {
                out.push_str(&format!(
                    "\nRead on: `job_read {{job: \"{job}\", page: {}}}`.",
                    page + 1
                ));
            } else if !s.ended && full {
                out.push_str(&format!(
                    "\nThe next page is being written: `job_read {{job: \"{job}\", page: {}, \
                     wait_s: 120}}` waits for it.",
                    page + 1
                ));
            } else if !s.ended {
                out.push_str(&format!(
                    "\nThis page is still being written: `job_read {{job: \"{job}\", page: {page}, \
                     wait_s: 120}}` waits for it to fill."
                ));
            }
            return Answer::ok(out);
        }
        pause(
            &mut wake,
            LOOK_EVERY.min(until.saturating_duration_since(Instant::now())),
        )
        .await;
    }
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    run(&OpsCall::of(ctx), args).await
}
