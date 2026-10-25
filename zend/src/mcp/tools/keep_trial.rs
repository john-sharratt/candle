//! `keep_trial {host, service, artifact}`: try a build on one machine through
//! its keep (`docs/claude_zend_gateway.md` §8.4) — keep's A/B trial, run as a
//! job: accepted at once, its outcome read with `keep_state`. keep refuses an
//! artifact not built exactly from a commit on `origin/main`, and promotes it
//! only once it serves, restoring the previous binary otherwise, so a bad build
//! cannot keep a member down. A Care operation, an Admin's.

use std::time::Duration;

use axum::http::Method;
use serde_json::{json, Value};

use super::{string, Answer, Ctx};
use crate::mcp::ops::{self, json_of};

pub const NAME: &str = "keep_trial";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Try a staged build on one machine through its keep: keep runs the artifact on the \
            service's stable path and promotes it once it prints its serving line, within ten minutes, or \
            puts the previous binary back. Answers at once; read the outcome with `keep_state`. Only a \
            build of a commit on origin/main is tried — a branch is tested on an agent. `artifact` is a \
            path on that machine, as the build job that staged it answered.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "The machine."},
                "service": {"type": "string", "enum": ["motherd", "web"]},
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
             ten minutes, or rolled back. Read the outcome with `keep_state {{host: \"{host}\"}}`."
        )),
        Err(e) => Answer::error(format!("keep on {host}: {e}")),
    }
}
