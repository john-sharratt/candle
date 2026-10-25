//! `cluster_command {member, name, args?, confirm?}`: run one of a member's
//! maintenance commands (`docs/claude_zend_gateway.md` §8.2). An Admin's.
//!
//! - **Classed before anything is sent** ([`class_of`]): a Look runs, a Care
//!   runs, a Decide is refused without the endpoint's `confirm` — the user's
//!   approval, which only Claude can pass on — and `offline`/`online` are done
//!   on the member's own machine, never through here.
//! - **The members' own checks stand behind it.** A destructive command's
//!   machine name, or a rollout's `count-hash`, goes in `args.confirm`, as the
//!   member asks for it.
//! - **`rollout` goes through the gateway**, which works out its plan: asked
//!   without `args.confirm` it answers the plan and changes nothing; with the
//!   build typed there, and the user's approval, it starts.

use std::time::Duration;

use axum::http::Method;
use host::commands::class::{class_of, Class};
use serde_json::{json, Value};

use super::{string, Answer, Ctx};
use crate::mcp::ops::{self, json_of, why};

pub const NAME: &str = "cluster_command";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Run one maintenance command on one zend member, named by its node as `cluster_status` \
            lists it. Answers its job, which `cluster_job` reads. Look commands (audit, self-check, \
            check-disk) and Care commands (drain, undrain, compact, maintain, expire-tombstones, catch-up, \
            resync, rejoin, release-leases, pause-rollout, resume-rollout) run as asked. Decide commands \
            (restart, shutdown, set-quorum, certify-anyway, reject-build, rollout, rollback, prune-builds, \
            upgrade-now) change what the cluster runs: ask the user first, and pass `confirm: \"approved by \
            the user\"` only with their approval for that command on that member. The member's own \
            arguments go in `args` — `app`, `build`, `from`, `max_members`, `builds`, `keep` — and a \
            destructive one's typed target (the machine's name; for rollout, the build) as `args.confirm`. \
            `rollout` without `args.confirm` answers the gateway's plan and changes nothing. An Admin's.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "member": {"type": "string", "description": "The member's node, 16 hex digits. Not used for rollout, which the gateway places."},
                "name": {"type": "string", "description": "The command."},
                "args": {"type": "object", "description": "The command's own arguments."},
                "confirm": {"type": "string", "description": "`approved by the user`, for a Decide command the user approved."}
            },
            "required": ["name"],
            "additionalProperties": false
        }
    })
}

/// Whether `name` may be sent with `confirm`: its class, or why not.
pub fn gate(name: &str, confirm: Option<&str>) -> Result<Class, String> {
    match class_of(name) {
        None => Err(format!("there is no maintenance command `{name}`")),
        Some(Class::OnTheMachine) => Err(format!(
            "`{name}` is done on the member's own machine: after `offline` it serves loopback \
             callers alone, so nothing could reach it through a gateway"
        )),
        Some(Class::Decide) => ops::approved(confirm, &format!("`{name}`")).map(|()| Class::Decide),
        Some(class) => Ok(class),
    }
}

/// A started job as the tool answers it.
pub fn render_started(member: &str, answer: &Value) -> String {
    format!(
        "Started `{}` on {member} as job {} — read it with `cluster_job {{member: \"{member}\", job: {}}}`.",
        answer.get("command").and_then(Value::as_str).unwrap_or("?"),
        answer.get("job").map_or_else(|| "?".into(), Value::to_string),
        answer.get("job").map_or_else(|| "?".into(), Value::to_string),
    )
}

async fn rollout(ctx: &Ctx, args: &Value, confirm: Option<&str>) -> Answer {
    let typed = args.get("confirm").and_then(Value::as_str);
    // The plan changes nothing; starting it is the decision.
    if typed.is_some() {
        if let Err(e) = ops::approved(confirm, "`rollout`") {
            return Answer::error(e);
        }
    }
    let reply = match ops::up(
        &ctx.session,
        &ctx.caller,
        Method::POST,
        "/gateway/upgrade/rollout",
        None,
        Some(args),
        Duration::from_secs(30),
    )
    .await
    {
        Ok(r) => r,
        Err(e) => return Answer::error(e),
    };
    match json_of(&reply) {
        Ok(v) if v.get("plan").is_some() && typed.is_none() => Answer::ok(format!(
            "The gateway's plan — nothing has changed:\n{}\n\nTo start it, with the user's approval, \
             call again with `args.confirm: {}` and `confirm: \"{}\"`.",
            serde_json::to_string_pretty(&v["plan"]).unwrap_or_default(),
            v.get("confirm").map_or_else(|| "the build".into(), Value::to_string),
            ops::APPROVED
        )),
        Ok(v) => Answer::ok(render_started(
            v.get("node").and_then(Value::as_str).unwrap_or("the plan's source"),
            &v,
        )),
        Err(e) => Answer::error(format!("rollout: {e}")),
    }
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    let name = match string(args, "name") {
        Ok(n) => n,
        Err(e) => return Answer::error(e),
    };
    let confirm = args.get("confirm").and_then(Value::as_str);
    let body = args.get("args").cloned().unwrap_or_else(|| json!({}));
    if !body.is_object() {
        return Answer::error("`args` is an object of the command's own arguments");
    }
    if name == "rollout" {
        return rollout(ctx, &body, confirm).await;
    }
    if let Err(e) = gate(name, confirm) {
        return Answer::error(e);
    }
    let member = match string(args, "member").and_then(ops::node) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    let reply = match ops::up(
        &ctx.session,
        &ctx.caller,
        Method::POST,
        &format!("/v1/cluster/command/{name}"),
        Some(&member),
        Some(&body),
        Duration::from_secs(30),
    )
    .await
    {
        Ok(r) => r,
        Err(e) => return Answer::error(e),
    };
    match reply.status {
        202 => match json_of(&reply) {
            Ok(v) => Answer::ok(render_started(&member, &v)),
            Err(e) => Answer::error(e),
        },
        409 => Answer::error(format!(
            "{member} is busy with another command: {} — read its job with `cluster_job`, then try \
             again",
            why(&reply)
        )),
        status => Answer::error(format!("`{name}` on {member}: {status} — {}", why(&reply))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A Decide command is refused without the user's approval**, a Look or
    /// a Care runs as asked, and the machine's own switch never goes through.
    #[test]
    fn every_command_is_gated_by_its_class() {
        assert_eq!(gate("audit", None), Ok(Class::Look));
        assert_eq!(gate("drain", None), Ok(Class::Care));
        assert!(gate("restart", None)
            .unwrap_err()
            .starts_with("`restart` is a decision"));
        assert_eq!(gate("restart", Some(ops::APPROVED)), Ok(Class::Decide));
        assert!(gate("online", Some(ops::APPROVED))
            .unwrap_err()
            .contains("done on the member's own machine"));
        assert_eq!(
            gate("frobnicate", None).unwrap_err(),
            "there is no maintenance command `frobnicate`"
        );
    }

    #[test]
    fn a_started_command_names_its_job_and_how_to_read_it() {
        assert_eq!(
            render_started("00000000000000ab", &json!({"job": 7, "command": "drain"})),
            "Started `drain` on 00000000000000ab as job 7 — read it with `cluster_job {member: \
             \"00000000000000ab\", job: 7}`."
        );
    }
}
