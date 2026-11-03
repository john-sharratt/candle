//! `job_cancel {job}`: stop a job, whatever its kind (`docs/jobs.md`): a build
//! killed at any phase, a deploy's build or its trial (rolled back), a task at
//! its next tool round. A job that has ended is left as it ended.

use serde_json::{json, Value};

use super::source::cancel;
use super::status::{allowed, job_schema};
use super::JobHandle;
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::tools::{string, Answer, Ctx};

pub const NAME: &str = "job_cancel";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Stop one job. A build or test run is killed at whatever phase it is in, and nothing it \
            built is staged. A deploy's build is killed and nothing is tried; a deploy keep is trying is \
            rolled back to the previous binary. A task stops at its next tool round and answers with what it \
            has; it is then stopped but live, and `repo_task_reply` redirects it. A maintenance job runs to \
            its end once started and cannot be stopped. A job that has ended is left as it ended.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job": job_schema()
            },
            "required": ["job"],
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
    cancel(call, &job).await
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    run(&OpsCall::of(ctx), args).await
}
