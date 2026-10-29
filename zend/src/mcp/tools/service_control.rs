//! `service_control {host, service, action, confirm?}`: restart, stop or start
//! a service through its machine's keep (`docs/claude_zend_gateway.md` §8.4).
//! A restart and a start are Care; a stop holds the service down until it is
//! started again, which is a decision the user approves. Admins only.

use std::time::Duration;

use axum::http::Method;
use serde_json::{json, Value};

use super::{string, Answer, Ctx};
use crate::mcp::ops::{self, json_of};

pub const NAME: &str = "service_control";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Restart, stop or start a service — motherd (the daemon that runs the models), web \
            (the gateway) or mcp (this connector's own server) — on one machine, through keep, the \
            supervisor that runs them. A restart runs the \
            installed binary again, with no trial; a start lifts a stop. A stop holds the service down until \
            it is started again — a decision: ask the user, and pass `confirm: \"approved by the user\"` only \
            with their approval. Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "The machine, as `cluster_status` names it."},
                "service": {"type": "string", "enum": ["motherd", "web", "mcp"]},
                "action": {"type": "string", "enum": ["restart", "stop", "start"]},
                "confirm": {"type": "string", "description": "`approved by the user`, for a stop."}
            },
            "required": ["host", "service", "action"],
            "additionalProperties": false
        }
    })
}

/// Refuse an action that is not one, or a stop the user has not approved.
pub fn gate(action: &str, confirm: Option<&str>) -> Result<(), String> {
    match action {
        "restart" | "start" => Ok(()),
        "stop" => ops::approved(confirm, "stopping a service"),
        other => Err(format!("`{other}` is not restart, stop or start")),
    }
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    let (host, service, action) = match (
        string(args, "host"),
        string(args, "service"),
        string(args, "action"),
    ) {
        (Ok(h), Ok(s), Ok(a)) => (h, s, a),
        (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => return Answer::error(e),
    };
    if let Err(e) = gate(action, args.get("confirm").and_then(Value::as_str)) {
        return Answer::error(e);
    }
    let reply = ops::to_keep(
        &ctx.session,
        &ctx.caller,
        Method::POST,
        &format!("/v1/keep/service/{service}/{action}"),
        host,
        None,
        Duration::from_secs(120),
    )
    .await;
    match reply.and_then(|r| json_of(&r)) {
        Ok(v) => Answer::ok(format!(
            "{service} on {host}: {}",
            v.get("answer").and_then(Value::as_str).unwrap_or("done")
        )),
        Err(e) => Answer::error(format!("{service} on {host}: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stop_needs_the_users_approval_and_a_restart_does_not() {
        assert!(gate("restart", None).is_ok());
        assert!(gate("start", None).is_ok());
        assert!(gate("stop", None)
            .unwrap_err()
            .starts_with("stopping a service is a decision"));
        assert!(gate("stop", Some(ops::APPROVED)).is_ok());
        assert!(gate("kill", None).is_err());
    }
}
