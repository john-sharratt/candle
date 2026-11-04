//! `approval_grant {id, confirm}`: run a call a conversation of Zen's asked for
//! and could not make alone (`docs/zen_tool_reach.md` §5) — a task's or a
//! chat's, held as approval `id` — once, exactly as it was asked, as the
//! caller. Only with the user's approval: `confirm` is the endpoint's word for
//! it, as on every Decide tool. Admins only.

use serde_json::{json, Value};

use super::{string, Answer, Ctx, LOADING};
use crate::mcp::approvals;
use crate::mcp::ops;
use crate::mcp::ops_tools::{run_granted, OpsCall};
use crate::mcp::workers::now_ms;

pub const NAME: &str = "approval_grant";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Grant a call a worker or chat of Zen's asked for and could not make alone — a \
            Decide command, a service stop, a timeline delete, a deploy the guard refused — held as an \
            approval: its answer said `needs_approval <id>`. Ask the user first: pass `confirm: \"approved \
            by the user\"` only with their approval for that call. It runs once, exactly as recorded, as \
            you, and answers what the call answered; pass that to the worker with `repo_task_reply`. An \
            approval left a day unsettled expires. Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "minLength": 12, "maxLength": 12, "description": "The approval's id, from the worker's `needs_approval` answer."},
                "confirm": {"type": "string", "description": "`approved by the user`."}
            },
            "required": ["id", "confirm"],
            "additionalProperties": false
        }
    })
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    let id = match string(args, "id") {
        Ok(id) => id,
        Err(e) => return Answer::error(e),
    };
    if let Err(e) = ops::approved(
        args.get("confirm").and_then(Value::as_str),
        &format!("granting approval {id}"),
    ) {
        return Answer::error(e);
    }
    let Some(records) = ctx.session.shared_substrate() else {
        return Answer::error(LOADING);
    };
    let approval = match approvals::settle(&records, &ctx.caller.key(), id, true, now_ms()) {
        Ok(a) => a,
        Err(e) => return Answer::error(e),
    };
    let answer = run_granted(&approval, &OpsCall::of(ctx)).await;
    let said = format!(
        "Approval {id} granted: `{}` ran as asked by {}, and answered:\n{}",
        approval.tool, approval.conversation, answer.text
    );
    if answer.is_error {
        Answer::error(said)
    } else {
        Answer::ok(said)
    }
}
