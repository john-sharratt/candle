//! `timeline_delete {id, confirm}`: tombstone one conversation in the
//! substrate (`docs/claude_zend_gateway.md` §8.5) — `DELETE
//! /v1/substrate/zend/timeline/:tl`, done in process and replicated as every
//! substrate write is. The supported repair for a conversation whose records
//! are intact and whose content is wrong: the next ingest pass rebuilds it. A
//! decision, refused without the user's approval; Admins only.

use serde_json::{json, Value};

use super::{Answer, Ctx, LOADING};
use crate::mcp::ops;

pub const NAME: &str = "timeline_delete";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Delete one stored conversation — a timeline, by the id `layer_read` lists — on every \
            member. The repair for a conversation whose content is wrong: one a layer's analysis built is \
            built again by the next analysis pass. Cannot be undone. A decision: ask the user, and pass \
            `confirm: \"approved by the user\"` only with their approval for that timeline. Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "The timeline id, a decimal number."},
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
    let Some(raw) = args
        .get("id")
        .and_then(|v| {
            v.as_str()
                .map(str::to_string)
                .or_else(|| v.as_u64().map(|n| n.to_string()))
        })
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|n| *n != 0)
    else {
        return Answer::error("`id` is a timeline id, a decimal number as `layer_read` lists it");
    };
    if let Err(e) = ops::approved(
        args.get("confirm").and_then(Value::as_str),
        &format!("deleting timeline {raw}"),
    ) {
        return Answer::error(e);
    }
    let session = ctx.session.clone();
    let done = tokio::task::spawn_blocking(move || session.tombstone_timeline_raw(raw)).await;
    match done {
        Ok(Some(Ok(()))) => Answer::ok(format!("Timeline {raw} is tombstoned.")),
        Ok(Some(Err(e))) => Answer::error(format!("timeline {raw} could not be tombstoned: {e}")),
        Ok(None) => Answer::error(LOADING),
        Err(e) => Answer::error(format!("the delete stopped: {e}")),
    }
}
