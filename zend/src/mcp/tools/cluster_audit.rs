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

/// How far back into the gateway's log a read reaches for operations: the log
/// also carries replication's own steps — a consensus round's prepare and
/// promise, a key claimed and released — many to the second, so the
/// operations among them are found deep in it.
const SCAN: u64 = 20_000;

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Read what was done to the cluster, newest first, from the gateway's log of it: each \
            maintenance command a member ran, its arguments, who sent it, and how it ended, with members \
            joining, leaving, being admitted or refused, and floors raised, beside them. Replication's own \
            steps, which the log also keeps, are left out. Use it before doing more, to see what has already \
            been done and by whom. Kept for a day. Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "entries": {"type": "integer", "minimum": 1, "maximum": MAX_ENTRIES, "description": "How many entries (50 by default)."}
            },
            "additionalProperties": false
        }
    })
}

/// Whether a log entry is an operation — about a member (its node id), or
/// the group as a whole — rather than a step of replicating one key, whose
/// subject is the key.
fn is_operation(entry: &Value) -> bool {
    let subject = entry.get("subject").and_then(Value::as_str).unwrap_or("");
    subject.is_empty() || (subject.len() == 16 && subject.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// The newest `n` operations in the log, newest first, one line each.
pub fn render(status: &Value, n: usize, now: u64) -> String {
    let entries: Vec<&Value> = status
        .get("log")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|e| is_operation(e))
        .take(n)
        .collect();
    if entries.is_empty() {
        return "The coordination log holds no operation from the last day.".into();
    }
    entries
        .iter()
        .map(|e| {
            let at = e.get("at_ms").and_then(Value::as_u64).unwrap_or(0);
            let field = |k: &str| e.get(k).and_then(Value::as_str).unwrap_or("");
            let about: Vec<&str> = ["gateway", "app", "subject"]
                .into_iter()
                .map(field)
                .filter(|s| !s.is_empty())
                .collect();
            format!("{} — {}: {}", ago(at, now), about.join(" "), field("text"))
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
        &format!("/gateway/status.json?log={SCAN}"),
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
        Ok(v) => Answer::ok(render(&v, n as usize, now_ms())),
        Err(e) => Answer::error(format!("the coordination log: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Operations, newest first, one line each** — replication's steps on
    /// a key are left out, and the count asked for is of operations.
    #[test]
    fn each_operation_is_one_line_newest_first_without_replication() {
        let now = 1_800_000_000_000;
        let status = json!({"log": [
            {"at_ms": now - 30_000, "gateway": "bw", "app": "npcd", "subject": "world/w1/entity/7",
             "text": "prepared at 3.1 by a0ac88178cb2b57d"},
            {"at_ms": now - 60_000, "gateway": "bw", "app": "zend", "subject": "a0ac88178cb2b57d",
             "text": "drain by j@x.com: ok"},
            {"at_ms": now - 90_000, "gateway": "bw", "app": "npcd", "subject": "lease/w1",
             "text": "claimed by a0ac88178cb2b57d at 9"},
            {"at_ms": now - 7_200_000, "gateway": "bw", "app": "zend", "subject": "",
             "text": "floor 3825 (bc8c1817) by certification"},
            {"at_ms": now - 7_300_000, "gateway": "bw", "app": "zend", "subject": "0000000000000002",
             "text": "joined"}
        ]});
        assert_eq!(
            render(&status, 2, now),
            "60 s ago — bw zend a0ac88178cb2b57d: drain by j@x.com: ok\n\
             2.0 h ago — bw zend: floor 3825 (bc8c1817) by certification"
        );
        assert_eq!(
            render(&json!({"log": []}), 50, now),
            "The coordination log holds no operation from the last day."
        );
    }
}
