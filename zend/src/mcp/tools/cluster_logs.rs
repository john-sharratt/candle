//! `cluster_logs`: what the members' logs say (`docs/claude_zend_gateway.md`
//! §8.1) — `GET /v1/logs`, which the gateway asks every member and gathers;
//! with `node`, one member's, which is for Admins only.

use std::time::Duration;

use axum::http::Method;
use serde_json::{json, Value};
use url::form_urlencoded::Serializer;

use super::{Answer, Ctx};
use crate::mcp::ops::{self, json_of};

pub const NAME: &str = "cluster_logs";

/// Lines answered when the call names no `limit`, and the most, as the
/// members' search allows.
const DEFAULT_LIMIT: u64 = 200;
const MAX_LIMIT: u64 = 5000;

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Search the logs of the cluster's members — the machines running the motherd daemon — for \
            what they say about a symptom: every member's matching lines, oldest first, each under its member. \
            `grep` is a regular expression found anywhere in a line; narrow with `level`, `tenant` and \
            `since`. A page, never a stream: call again with a later `since` to read on. Cite the lines you \
            rely on.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "grep": {"type": "string", "description": "A regular expression; a line is kept when it is found anywhere in it."},
                "level": {"type": "string", "enum": ["TRACE", "DEBUG", "INFO", "WARN", "ERROR"], "description": "The least level kept."},
                "tenant": {"type": "string", "enum": ["zend", "npcd"], "description": "Only lines one app wrote: `zend`, the coding service these tools run on, or `npcd`, the bots'."},
                "since": {"type": "string", "description": "Only lines stamped at or after this: an RFC 3339 prefix such as `2026-10-16T00:20`."},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT, "description": "The most lines per member, the latest kept (200 by default)."},
                "node": {"type": "string", "description": "One member's node id, as `cluster_status` lists it (Admins only)."}
            },
            "additionalProperties": false
        }
    })
}

/// The `/v1/logs` request `args` asks for.
pub fn path_of(args: &Value) -> String {
    let mut q = Serializer::new(String::new());
    for key in ["grep", "level", "tenant", "since"] {
        if let Some(v) = args.get(key).and_then(Value::as_str) {
            q.append_pair(key, v);
        }
    }
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_LIMIT)
        .clamp(1, MAX_LIMIT);
    q.append_pair("limit", &limit.to_string());
    format!("/v1/logs?{}", q.finish())
}

/// Every member's lines, under its member.
pub fn render(envelope: &Value) -> String {
    let mut out = String::new();
    for m in envelope
        .get("members")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let host = m.get("host").and_then(Value::as_str).unwrap_or("?");
        let node = m.get("node").and_then(Value::as_str).unwrap_or("?");
        let code = m.get("status").and_then(Value::as_u64).unwrap_or(0);
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        if code != 200 {
            out.push_str(&format!("{host} {node}: no answer ({code})"));
            continue;
        }
        let body = m.get("body").cloned().unwrap_or(Value::Null);
        let lines: Vec<&str> = body
            .get("lines")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        let matched = body.get("matched").and_then(Value::as_u64).unwrap_or(0);
        out.push_str(&format!(
            "{host} {node}: {} of {matched} matching lines",
            lines.len()
        ));
        for line in lines {
            out.push('\n');
            out.push_str(line);
        }
    }
    if out.is_empty() {
        "No member answered.".into()
    } else {
        out
    }
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    let node = match args.get("node").and_then(Value::as_str) {
        Some(n) => match ops::admin_only(&ctx.caller, "cluster_logs` with `node")
            .and_then(|()| ops::node(n))
        {
            Ok(n) => Some(n),
            Err(e) => return Answer::error(e),
        },
        None => None,
    };
    let reply = match ops::up(
        &ctx.session,
        &ctx.caller,
        Method::GET,
        &path_of(args),
        node.as_deref(),
        None,
        Duration::from_secs(30),
    )
    .await
    {
        Ok(r) => r,
        Err(e) => return Answer::error(e),
    };
    match json_of(&reply) {
        // One member, named: its own answer, not an envelope.
        Ok(v) if v.get("members").is_none() => Answer::ok(render(&json!({
            "members": [{"host": "", "node": node.unwrap_or_default(), "status": 200, "body": v}]
        }))),
        Ok(v) => Answer::ok(render(&v)),
        Err(e) => Answer::error(format!("/v1/logs: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_query_carries_what_was_asked_encoded() {
        assert_eq!(
            path_of(&json!({"grep": "a b|c", "level": "WARN", "limit": 9_999})),
            "/v1/logs?grep=a+b%7Cc&level=WARN&limit=5000"
        );
        assert_eq!(path_of(&json!({})), "/v1/logs?limit=200");
    }

    #[test]
    fn each_members_lines_come_under_it() {
        let envelope = json!({"members": [
            {"host": "bw", "node": "01", "status": 200,
             "body": {"lines": ["2026 WARN x", "2026 ERROR y"], "matched": 5}},
            {"host": "ww", "node": "02", "status": 504}
        ]});
        assert_eq!(
            render(&envelope),
            "bw 01: 2 of 5 matching lines\n2026 WARN x\n2026 ERROR y\n\nww 02: no answer (504)"
        );
        assert_eq!(render(&json!({"members": []})), "No member answered.");
    }
}
