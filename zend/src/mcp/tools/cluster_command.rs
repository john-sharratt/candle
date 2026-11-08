//! `cluster_command {member, name, args?, confirm?}`: run one of a member's
//! maintenance commands (`docs/claude_zend_gateway.md` §8.2). Admins only.
//!
//! The commands it names and offers are the members' own table
//! (`host::commands::class::COMMANDS`), so a command added there is offered
//! here, in its class, with nothing to keep in step.
//!
//! - **Classed before anything is sent** ([`class_of`]): a Look runs, a Care
//!   runs, a Decide is refused without the endpoint's `confirm` — the user's
//!   approval, which only the caller can pass on — and `offline`/`online` are done
//!   on the member's own machine, never through here.
//! - **The members' own checks stand behind it.** A destructive command's
//!   machine name, or a rollout's `count-hash`, goes in `args.confirm`, as the
//!   member asks for it.
//! - **`rollout` goes through the gateway**, which works out its plan: asked
//!   without `args.confirm` it answers the plan and changes nothing; with the
//!   build typed there, and the user's approval, it starts.

use std::time::Duration;

use axum::http::Method;
use host::commands::class::{class_of, Class, COMMANDS};
use serde_json::{json, Value};

use super::job::JobHandle;
use super::{string, Answer, Ctx};
use crate::mcp::ops::{self, json_of, why};
use crate::mcp::ops_tools::OpsCall;

pub const NAME: &str = "cluster_command";

/// The commands of `class`, by name, as the members define them.
fn named(class: Class) -> Vec<&'static str> {
    COMMANDS
        .iter()
        .filter(|(_, c)| *c == class)
        .map(|(n, _)| *n)
        .collect()
}

pub fn definition() -> Value {
    let (look, care, decide) = (named(Class::Look), named(Class::Care), named(Class::Decide));
    let sendable: Vec<&str> = look.iter().chain(&care).chain(&decide).copied().collect();
    json!({
        "name": NAME,
        "description": format!(
            "Run one maintenance command on one member of the cluster — a machine running the motherd daemon — \
            named by its node id as `cluster_status` lists it. Answers the job it started, \
            `cluster:<node>:<n>`: wait on it with `job_status` and read its steps with `job_read`. Commands come in three kinds. Look commands only read ({}) and Care \
            commands are routine upkeep ({}): both run as asked. Decide commands change what the cluster \
            runs ({}): ask the user first, and pass `confirm: \"approved by the user\"` only with their \
            approval for that command on that member. A command's own arguments go in `args` — `app`, \
            `build`, `from`, `max_members`, `builds`, `keep`, and `newest` for an `audit` that samples \
            only the newest that many segments rather than walking them all — and a destructive one's typed target (the \
            machine's name; for `rollout`, the build) as `args.confirm`. `rollout` without `args.confirm` \
            answers the plan and changes nothing. Admins only.",
            look.join(", "),
            care.join(", "),
            decide.join(", "),
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "member": {"type": "string", "description": "The member's node id, 16 hex digits. Not used for `rollout`, which the gateway places."},
                "name": {"type": "string", "enum": sendable, "description": "The command."},
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
    let command = answer.get("command").and_then(Value::as_str).unwrap_or("?");
    match answer.get("job").and_then(Value::as_u64) {
        Some(n) => {
            let job = JobHandle::Cluster {
                node: member.to_string(),
                n,
            };
            format!(
                "Started {job}: `{command}` on {member}. `job_status {{jobs: [\"{job}\"], wait_s: \
                 240}}` waits for it to end and says how it went."
            )
        }
        None => format!("`{command}` on {member} answered no job: {answer}"),
    }
}

/// What a call of this tool with `args` is: Look, Care or Decide — a rollout
/// with its build typed in `args.confirm` starts one, and without it only
/// answers the plan — or why it cannot be sent at all.
pub fn class_of_call(args: &Value) -> Result<Class, String> {
    let name = string(args, "name")?;
    if name == "rollout" {
        let typed = args.pointer("/args/confirm").and_then(Value::as_str);
        return Ok(if typed.is_some() {
            Class::Decide
        } else {
            Class::Look
        });
    }
    match class_of(name) {
        None => Err(format!("there is no maintenance command `{name}`")),
        Some(Class::OnTheMachine) => gate(name, None).map(|_| Class::OnTheMachine),
        Some(class) => Ok(class),
    }
}

async fn rollout(call: &OpsCall, args: &Value, confirm: Option<&str>) -> Answer {
    let typed = args.get("confirm").and_then(Value::as_str);
    // The plan changes nothing; starting it is the decision.
    if typed.is_some() {
        if let Err(e) = ops::approved(confirm, "`rollout`") {
            return Answer::error(e);
        }
    }
    let reply = match ops::up_as(
        &call.session,
        call.headers(),
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
    run(&OpsCall::of(ctx), args).await
}

/// The command for `call`, whose caller is checked an Admin: a Decide one only
/// with `confirm` (`docs/claude_zend_gateway.md` §8.2).
pub async fn run(call: &OpsCall, args: &Value) -> Answer {
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
        return rollout(call, &body, confirm).await;
    }
    if let Err(e) = gate(name, confirm) {
        return Answer::error(e);
    }
    let member = match string(args, "member").and_then(ops::node) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    let reply = match ops::up_as(
        &call.session,
        call.headers(),
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
            "{member} is busy with another command: {} — `job_list {{kind: \"cluster\"}}` names \
             its job; wait on it with `job_status`, then try again",
            why(&reply)
        )),
        status => Answer::error(format!("`{name}` on {member}: {status} — {}", why(&reply))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_def;

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

    /// **Zen's `cluster_command` offers exactly the commands this tool sends**:
    /// its YAML's enum is the members' own table less the machine's switches,
    /// so a command added there is a test failure here until Zen offers it.
    #[test]
    fn zens_definition_offers_the_commands_this_tool_sends() {
        let zen = tool_def::find(NAME).expect("cluster_command is one of Zen's tools");
        let mut offered: Vec<String> = zen.parameters["properties"]["name"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        let mut sent: Vec<String> = definition()["inputSchema"]["properties"]["name"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        offered.sort();
        sent.sort();
        assert_eq!(offered, sent);
    }

    /// **A call is classed by its command**, and a rollout by whether its
    /// build is typed: the plan only reads, the start is the decision.
    #[test]
    fn a_call_is_classed_by_its_command() {
        assert_eq!(class_of_call(&json!({"name": "audit"})), Ok(Class::Look));
        assert_eq!(class_of_call(&json!({"name": "drain"})), Ok(Class::Care));
        assert_eq!(
            class_of_call(&json!({"name": "restart"})),
            Ok(Class::Decide)
        );
        assert_eq!(
            class_of_call(&json!({"name": "rollout", "args": {"app": "zend"}})),
            Ok(Class::Look)
        );
        assert_eq!(
            class_of_call(&json!({"name": "rollout", "args": {"confirm": "4419-6ecd6158"}})),
            Ok(Class::Decide)
        );
        assert!(class_of_call(&json!({"name": "offline"})).is_err());
        assert!(class_of_call(&json!({"name": "frobnicate"})).is_err());
    }

    #[test]
    fn a_started_command_names_its_job_and_how_to_read_it() {
        assert_eq!(
            render_started("00000000000000ab", &json!({"job": 7, "command": "drain"})),
            "Started cluster:00000000000000ab:7: `drain` on 00000000000000ab. `job_status {jobs: \
             [\"cluster:00000000000000ab:7\"], wait_s: 240}` waits for it to end and says how it \
             went."
        );
    }

    /// **The tool offers every command a member runs through the gateway, and
    /// nothing else**: the machine's own switches are never in its enum, and
    /// every Decide command is named as one in the description.
    #[test]
    fn the_offered_commands_are_the_members_own_less_the_machines_switches() {
        let def = definition();
        let offered: Vec<&str> = def["inputSchema"]["properties"]["name"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(offered.len(), COMMANDS.len() - 2);
        assert!(!offered.contains(&"offline") && !offered.contains(&"online"));
        for name in offered {
            assert!(gate(name, Some(ops::APPROVED)).is_ok(), "{name}");
        }
        let text = def["description"].as_str().unwrap();
        assert!(text.contains(&named(Class::Decide).join(", ")), "{text}");
    }
}
