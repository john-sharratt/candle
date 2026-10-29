//! `service_trial {host, service, artifact}`: try a build of a service on one
//! machine through its keep (`docs/claude_zend_gateway.md` §8.4) — keep's A/B
//! trial, run as a job: accepted at once, its outcome read with
//! `service_status`. keep refuses an artifact not built exactly from a commit
//! on `origin/main`, and promotes it only once it serves, restoring the
//! previous binary otherwise, so a bad build cannot keep a member down. A Care
//! operation, Admins only.

use std::time::Duration;

use axum::http::Method;
use serde_json::{json, Value};

use super::{string, Answer, Ctx};
use crate::mcp::ops::{self, json_of};

pub const NAME: &str = "service_trial";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Try a new build of a service — motherd, web (the gateway) or mcp (this connector's \
            own server, which serves the guides) — on one machine. keep, the supervisor \
            that runs the machine's services, starts the build in place of the installed one and keeps it \
            once it reports it is serving, within ten minutes; otherwise it puts the previous binary back. \
            Answers at once; read the outcome with `service_status`. Only a build of a commit on origin/main \
            is tried — test a branch with `repo_job_start`. `artifact` is the build's path on that machine, as \
            `repo_job_log` answered it for the job that staged it. Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "The machine, as `cluster_status` names it."},
                "service": {"type": "string", "enum": ["motherd", "web", "mcp"]},
                "artifact": {"type": "string", "description": "The staged executable's path on that machine."}
            },
            "required": ["host", "service", "artifact"],
            "additionalProperties": false
        }
    })
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    let (host, service, artifact) = match (
        string(args, "host"),
        string(args, "service"),
        string(args, "artifact"),
    ) {
        (Ok(h), Ok(s), Ok(a)) => (h, s, a),
        (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => return Answer::error(e),
    };
    let reply = ops::to_keep(
        &ctx.session,
        &ctx.caller,
        Method::POST,
        "/v1/keep/trial",
        host,
        Some(&json!({"service": service, "artifact": artifact})),
        Duration::from_secs(60),
    )
    .await;
    match reply.and_then(|r| json_of(&r)) {
        Ok(_) => Answer::ok(format!(
            "keep on {host} is trying {artifact} as {service}: it is proven once it serves, within \
             ten minutes, or rolled back. Read the outcome with `service_status {{host: \"{host}\"}}`."
        )),
        Err(e) => Answer::error(format!("keep on {host}: {e}")),
    }
}
