//! `repo_job_cancel {job}`: stop a build-agent job at whatever phase it is in
//! — its fetch abandoned, its place in the checkout's queue given up, or its
//! command and everything it started killed. A job that has already ended is
//! left as it ended, and the answer says how.

use serde_json::{json, Value};

use super::{describe, host_of, lost, Handle, NO_AGENTS};
use crate::mcp::ops;
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::tools::{string, Answer, Ctx};
use crate::mcp::workers::now_ms;

pub const NAME: &str = "repo_job_cancel";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Stop a build-agent job, whatever it is doing: one still fetching origin or waiting \
            for its checkout never runs, and a running one is killed with everything it started; nothing is \
            staged. The checkout is put back as it was. A job that has already ended is left as it ended, \
            and the answer says how. Cancelling a deploy's build ends the deploy; to stop a deploy whose \
            trial has begun, use `service_cancel`. `job` is the handle, `<node>:<id>`. Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job": {"type": "string", "minLength": 18, "maxLength": 64, "description": "The job's handle, `<node>:<id>`."}
            },
            "required": ["job"],
            "additionalProperties": false
        }
    })
}

pub async fn run(call: &OpsCall, args: &Value) -> Answer {
    let h = match string(args, "job").and_then(Handle::parse) {
        Ok(h) => h,
        Err(e) => return Answer::error(e),
    };
    let Some(agents) = call.session.agents() else {
        return Answer::error(NO_AGENTS);
    };
    let at = h.at(&agents);
    let before = match agents.view(at, &h.id).await {
        Ok(v) => v,
        Err(e) => return Answer::error(lost(&h, &e)),
    };
    let host = host_of(&agents, h.node).await;
    if before.status.is_finished() {
        return Answer::ok(format!(
            "Nothing to cancel: it had already ended.\n{}",
            describe(&h, &host, &before, now_ms())
        ));
    }
    match agents.cancel(at, &h.id).await {
        Ok(view) => Answer::ok(format!(
            "Cancelled.\n{}",
            describe(&h, &host, &view, now_ms())
        )),
        Err(e) => Answer::error(lost(&h, &e)),
    }
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    run(&OpsCall::of(ctx), args).await
}
