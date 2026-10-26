//! `cluster_audit`: who ran which command, and how it ended
//! (`docs/claude_zend_gateway.md` §8.1) — the gateway's coordination log, which
//! records every command a member ran and who sent it, kept in the gateway's
//! memory for a day. Admins only.

use std::time::Duration;

use axum::http::Method;
use serde_json::{json, Value};

use super::cluster_status::ago;
use super::{Answer, Ctx};
use crate::mcp::ops::{self, json_of};
use crate::mcp::workers::now_ms;

pub const NAME: &str = "cluster_audit";

const DEFAULT_ENTRIES: u64 = 50;
const MAX_ENTRIES: u64 = 500;

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Read what was done to the cluster, newest first, from the gateway's log of it: each \
            maintenance command a member ran, its arguments, who sent it, and how it ended, with members \
            joining and leaving beside them. Use it before doing more, to see what has already been done and \
            by whom. Kept for a day. Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "entries": {"type": "integer", "minimum": 1, "maximum": MAX_ENTRIES, "description": "How many entries (50 by default)."}
            },
            "additionalProperties": false
        }
    })
}

/// The log's entries, newest first, one line each.
pub fn render(status: &Value, now: u64) -> String {
    let entries = status
        .get("log")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if entries.is_empty() {
        return "The coordination log holds nothing from the last day.".into();
    }
    entries
        .iter()
        .map(|e| {
            let at = e.get("at_ms").and_then(Value::as_u64).unwrap_or(0);
            let field = |k: &str| e.get(k).and_then(Value::as_str).unwrap_or("").to_string();
            format!(
                "{} — {} {} {}: {}",
                ago(at, now),
                field("gateway"),
                field("app"),
                field("subject"),
                field("text")
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    let n = args
        .get("entries")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_ENTRIES)
        .clamp(1, MAX_ENTRIES);
    let reply = match ops::up(
        &ctx.session,
        &ctx.caller,
        Method::GET,
        &format!("/gateway/status.json?log={n}"),
        None,
        None,
        Duration::from_secs(15),
    )
    .await
    {
        Ok(r) => r,
        Err(e) => return Answer::error(e),
    };
    match json_of(&reply) {
        Ok(v) => Answer::ok(render(&v, now_ms())),
        Err(e) => Answer::error(format!("the coordination log: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_entry_is_one_line_newest_first() {
        let now = 1_800_000_000_000;
        let status = json!({"log": [
            {"at_ms": now - 60_000, "gateway": "bw", "app": "zend", "subject": "01",
             "text": "drain by j@x.com: ok"},
            {"at_ms": now - 7_200_000, "gateway": "bw", "app": "zend", "subject": "02",
             "text": "joined"}
        ]});
        assert_eq!(
            render(&status, now),
            "60 s ago — bw zend 01: drain by j@x.com: ok\n2.0 h ago — bw zend 02: joined"
        );
        assert_eq!(
            render(&json!({"log": []}), now),
            "The coordination log holds nothing from the last day."
        );
    }
}
