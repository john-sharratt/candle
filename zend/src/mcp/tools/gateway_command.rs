//! `gateway_command {name, args?, confirm?}`: one of the gateway's own commands
//! (`docs/claude_zend_gateway.md` §8.2) — `raise-floor`, `cancel-floor`. Each is
//! a decision, refused without the user's approval. It reaches the gateway this
//! member's socket is on. Admins only.
//!
//! The gateway's drain is not one: draining closes the very socket this tool
//! reaches the gateway by, and a drained gateway admits none for an undrain to
//! come back up, so both are done from the gateway's page or its machine.

use std::time::Duration;

use axum::http::Method;
use serde_json::{json, Value};

use super::{string, Answer, Ctx};
use crate::mcp::ops::{self, json_of};

pub const NAME: &str = "gateway_command";

/// Each command and the route it is.
pub fn route_of(name: &str) -> Option<String> {
    match name {
        "raise-floor" | "cancel-floor" => Some(format!("/gateway/upgrade/{name}")),
        _ => None,
    }
}

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Run one of the gateway's own upgrade commands, for an app (`zend`, the coding service, or `npcd`, the bots') whose \
            members are moving to a new build. During an upgrade the gateway keeps routing to members on the \
            older build for a grace period, until the app's floor — the oldest build it still routes to — \
            rises. `raise-floor` raises it now, ending the grace period; `cancel-floor` drops a pending rise. \
            Pass `args: {app}`. Each is a decision: ask the user, and pass `confirm: \"approved by the user\"` \
            only with their approval. Draining a gateway is not a tool — it would cut this connection — and \
            is done from the gateway's page. Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": ["raise-floor", "cancel-floor"]},
                "args": {"type": "object", "description": "`{app}` for the floor commands."},
                "confirm": {"type": "string", "description": "`approved by the user`."}
            },
            "required": ["name"],
            "additionalProperties": false
        }
    })
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    let name = match string(args, "name") {
        Ok(n) => n,
        Err(e) => return Answer::error(e),
    };
    let Some(path) = route_of(name) else {
        return Answer::error(format!("there is no gateway command `{name}`"));
    };
    if let Err(e) = ops::approved(
        args.get("confirm").and_then(Value::as_str),
        &format!("`{name}`"),
    ) {
        return Answer::error(e);
    }
    let body = args.get("args").cloned().unwrap_or_else(|| json!({}));
    let reply = match ops::up(
        &ctx.session,
        &ctx.caller,
        Method::POST,
        &path,
        None,
        Some(&body),
        Duration::from_secs(30),
    )
    .await
    {
        Ok(r) => r,
        Err(e) => return Answer::error(e),
    };
    match json_of(&reply) {
        Ok(v) => Answer::ok(format!("`{name}`: {v}")),
        Err(e) => Answer::error(format!("`{name}`: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_command_is_its_route_and_nothing_else_is_one() {
        assert_eq!(
            route_of("raise-floor").unwrap(),
            "/gateway/upgrade/raise-floor"
        );
        assert_eq!(
            route_of("cancel-floor").unwrap(),
            "/gateway/upgrade/cancel-floor"
        );
        assert_eq!(route_of("drain-gateway"), None);
        assert_eq!(route_of("undrain-gateway"), None);
        assert_eq!(route_of("rollout"), None);
    }
}
