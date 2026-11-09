//! `job_status {jobs, wait_s?}`: where each job is, waiting up to `wait_s` for
//! any of them to end (`docs/jobs.md` §4). One call that holds, so a caller
//! starts every job first and waits on them together; each step is said on the
//! call's progress while it waits.

use std::time::Duration;

use serde_json::{json, Value};
use tokio::time::Instant;

use super::source::standing;
use super::wait::wait_on;
use super::{pages, JobHandle, Standing};
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::tools::{Answer, Ctx};

pub const NAME: &str = "job_status";

/// The most jobs one call waits on.
const MAX_JOBS: usize = 16;

/// The seconds a call waits when it names no `wait_s`.
const DEFAULT_WAIT_S: u64 = 60;

/// The handle pattern every `job` takes (`docs/jobs.md` §7).
pub const HANDLE_PATTERN: &str = "(task:[a-z0-9-]+/[a-z0-9-]+([.][0-9]+)?|(build|deploy):[0-9a-f]{16}:[A-Za-z0-9_-]{10}[AEIMQUYcgkosw048]|cluster:[0-9a-f]{16}:[0-9]{1,10})";

/// The task part of [`HANDLE_PATTERN`]: what a tool that steers a task takes.
pub const TASK_PATTERN: &str = "task:[a-z0-9-]+/[a-z0-9-]+([.][0-9]+)?";

/// What a tool that steers a task says of the `task` it takes.
pub fn task_schema() -> Value {
    json!({
        "type": "string", "minLength": 8, "maxLength": 200, "pattern": TASK_PATTERN,
        "description": "The task's handle, `task:<thread>/<name>`, as `repo_task_start` or `job_list` answered it."
    })
}

/// What every tool says of the `job` it takes.
pub fn job_schema() -> Value {
    json!({
        "type": "string", "minLength": 8, "maxLength": 200, "pattern": HANDLE_PATTERN,
        "description": "The job's handle, as the call that started it or `job_list` answered it: `task:<thread>/<name>`, `build:<node>:<id>`, `deploy:<node>:<id>` or `cluster:<node>:<n>`."
    })
}

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "Say where jobs are — tasks, builds, deploys, maintenance jobs — and wait up to `wait_s` \
            seconds (60 by default; 0 answers at once) for any one of them to end. Give every job you are \
            waiting on in one call, and it answers as soon as the first ends; call it again for the rest. \
            For each: where it is, and once ended how it went — a build's or trial's test results and first \
            failures by line, a task's answer and the journal entries it has written since you last heard, \
            a maintenance job's result — then `pages: N` of output, which `job_read` pages and `job_grep` \
            searches. Waiting is one call, not a loop. A task asking a question waits for `repo_task_reply`.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "jobs": {"type": "array", "minItems": 1, "maxItems": MAX_JOBS, "items": job_schema(),
                         "description": "The jobs to wait on, by handle; the answer comes as soon as any one ends."},
                "wait_s": {"type": "integer", "minimum": 0, "maximum": 3600, "description": "Seconds to wait for one to end; 60 by default, 0 to answer at once. The caller's own limit caps it."}
            },
            "required": ["jobs"],
            "additionalProperties": false
        }
    })
}

/// The handles `args` names, or why it names none a caller may read.
pub fn handles_of(call: &OpsCall, args: &Value) -> Result<Vec<JobHandle>, String> {
    let named: Vec<&str> = args
        .get("jobs")
        .and_then(Value::as_array)
        .ok_or("`jobs` is a list of job handles")?
        .iter()
        .filter_map(Value::as_str)
        .collect();
    if named.is_empty() || named.len() > MAX_JOBS {
        return Err(format!("`jobs` names one to {MAX_JOBS} jobs"));
    }
    named
        .into_iter()
        .map(|n| {
            let h = JobHandle::parse(n)?;
            allowed(call, &h)?;
            Ok(h)
        })
        .collect()
}

/// Refuse a job of a kind only an Admin reads to a caller who is not one.
pub fn allowed(call: &OpsCall, job: &JobHandle) -> Result<(), String> {
    match job {
        JobHandle::Task { .. } => Ok(()),
        _ if call.admin() => Ok(()),
        _ => Err(format!(
            "{job} is a {}: reading one is an Admin's",
            job.kind()
        )),
    }
}

/// One job as an answer says it.
pub fn render(job: &JobHandle, s: &Standing) -> String {
    format!(
        "{}{}\npages: {} — `job_read {{job: \"{job}\", page: P}}` reads page P",
        s.line,
        s.detail,
        pages(s.total)
    )
}

pub async fn run(call: &OpsCall, args: &Value) -> Answer {
    let jobs = match handles_of(call, args) {
        Ok(j) => j,
        Err(e) => return Answer::error(e),
    };
    let (secs, asked) = call.wait_of(args, DEFAULT_WAIT_S);
    let until = Instant::now() + Duration::from_secs(secs);
    let mut said: Vec<String> = vec![String::new(); jobs.len()];
    // Subscribed before the first look, so a task's change between the look
    // and the wait still wakes it.
    let wake = call.session.mcp().watch();
    let standings = wait_on(
        jobs.len(),
        |i| matches!(jobs[i], JobHandle::Task { .. }),
        until,
        |i| standing(call, &jobs[i]),
        Some(wake),
        |i, s| {
            if s.line != said[i] {
                call.say(s.line.clone());
                said[i] = s.line.clone();
            }
        },
    )
    .await;
    let mut blocks = Vec::new();
    let mut marks = Vec::new();
    let mut running = Vec::new();
    for (job, s) in jobs.iter().zip(standings) {
        match s {
            Ok(s) => {
                if !s.ended {
                    running.push(job.to_string());
                }
                marks.extend(s.marks.iter().cloned());
                blocks.push(render(job, &s));
            }
            Err(e) => blocks.push(format!("{job}: {e}")),
        }
    }
    let mut out = blocks.join("\n\n");
    if !running.is_empty() {
        let cut = asked.map_or(String::new(), |a| {
            format!(" (you asked for {a} s; this caller allows {secs})")
        });
        out.push_str(&format!(
            "\n\nStill going after {secs} s{cut}: `job_status {{jobs: {}, wait_s: {}}}` keeps waiting.",
            json!(running),
            call.ceiling.min(600)
        ));
    }
    Answer::ok(out).marking(marks)
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    run(&OpsCall::of(ctx), args).await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The handle pattern admits each kind's exact shape, and nothing
    /// else.**
    #[test]
    fn the_handle_pattern_admits_each_kinds_shape() {
        let re = regex::Regex::new(&format!("^{HANDLE_PATTERN}$")).unwrap();
        for ok in [
            "task:review/verify",
            "task:review/verify.2",
            "build:76c3a0f4d39bd9db:XeV_I_QPzpg",
            "deploy:76c3a0f4d39bd9db:XeV_I_QPzpg",
            "cluster:a0ac88178cb2b57d:7",
        ] {
            assert!(re.is_match(ok), "{ok}");
        }
        for bad in [
            "task:review",
            "task:review/verify.",
            "task:review/verify/read",
            "task:review/verify.2/read.3",
            "build:76c3a0f4d39bd9db:XeV_I_QPzpZ",
            "build:76C3A0F4D39BD9DB:XeV_I_QPzpg",
            "cluster:a0ac88178cb2b57d:",
            "job:1",
            "the build",
        ] {
            assert!(!re.is_match(bad), "{bad}");
        }
        assert!(
            HANDLE_PATTERN.starts_with(&format!("({TASK_PATTERN}|")),
            "a task's pattern is the task part of every handle's"
        );
    }

    #[test]
    fn a_job_reads_as_its_line_its_detail_and_how_to_page_it() {
        let job = JobHandle::parse("build:00000000000000ab:XeV_I_QPzpg").unwrap();
        let s = Standing {
            ended: true,
            line: "build:00000000000000ab:XeV_I_QPzpg on BlackWorld: exited 0".into(),
            detail: "\ntest results:\n  9: test result: ok".into(),
            total: 450,
            size: 9_000,
            marks: Vec::new(),
        };
        assert_eq!(
            render(&job, &s),
            "build:00000000000000ab:XeV_I_QPzpg on BlackWorld: exited 0\ntest results:\n  9: test \
             result: ok\npages: 3 — `job_read {job: \"build:00000000000000ab:XeV_I_QPzpg\", page: \
             P}` reads page P"
        );
    }
}
