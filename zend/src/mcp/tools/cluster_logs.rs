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
/// members' search allows. The default keeps a page inside what a client
/// reads in one answer.
const DEFAULT_LIMIT: u64 = 100;
const MAX_LIMIT: u64 = 5000;

/// The most of one line shown: a line that dumps a structure runs to
/// kilobytes, and its head says what it is.
const LINE_CHARS: usize = 400;

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "Search the logs of the cluster's members — the machines running the motherd daemon — for \
            what they say about a symptom: every member's matching lines, oldest first, each under its member, \
            each cut at 400 characters. `grep` is a regular expression found anywhere in a line; narrow with \
            `level`, `tenant` and `since` (UTC). When more lines match than `limit`, the newest are kept and \
            the answer says earlier ones were left out: narrow the search, or name `since`, to see them. Cite \
            the lines you rely on.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "grep": {"type": "string", "description": "A regular expression; a line is kept when it is found anywhere in it."},
                "level": {"type": "string", "enum": ["TRACE", "DEBUG", "INFO", "WARN", "ERROR"], "description": "The least level kept."},
                "tenant": {"type": "string", "enum": ["zend", "npcd"], "description": "Only lines one app wrote: `zend`, the coding service these tools run on, or `npcd`, the bots'."},
                "since": {"type": "string", "description": "Only lines stamped at or after this: an RFC 3339 prefix such as `2026-10-16T00:20`."},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT, "description": "The most lines per member, the newest kept (100 by default)."},
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
        let left_out = matched.saturating_sub(lines.len() as u64);
        // A search that stopped at its limit read no further back: there may
        // be earlier matches it never counted.
        let complete = body
            .get("complete")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        if !complete {
            // How many it left out is not known: only that it stopped.
            out.push_str(&format!(
                "{host} {node}: {} matching lines — the newest; earlier ones may be left out \
                 (narrow the search, or name `since`, to see them)",
                lines.len()
            ));
        } else {
            out.push_str(&format!(
                "{host} {node}: {} of {matched} matching lines",
                lines.len()
            ));
            if left_out > 0 {
                out.push_str(&format!(
                    " — the newest; the {left_out} earlier are left out (narrow the search to see \
                     them)"
                ));
            }
        }
        for line in lines {
            out.push('\n');
            if line.chars().count() > LINE_CHARS {
                out.extend(line.chars().take(LINE_CHARS));
                out.push('…');
            } else {
                out.push_str(line);
            }
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
        assert_eq!(path_of(&json!({})), "/v1/logs?limit=100");
    }

    /// **Each member's lines come under it**, a long line cut, and lines
    /// left out by the limit counted and said to be the earlier ones.
    #[test]
    fn each_members_lines_come_under_it() {
        let long = format!("2026 WARN {}", "z".repeat(500));
        let envelope = json!({"members": [
            {"host": "bw", "node": "01", "status": 200,
             "body": {"lines": ["2026 WARN x", long], "matched": 5}},
            {"host": "ww", "node": "02", "status": 504},
            {"host": "aw", "node": "03", "status": 200,
             "body": {"lines": ["2026 ERROR y"], "matched": 1, "complete": true}},
            {"host": "lw", "node": "04", "status": 200,
             "body": {"lines": ["2026 INFO z"], "matched": 1, "complete": false}}
        ]});
        let cut: String = long.chars().take(LINE_CHARS).collect();
        assert_eq!(
            render(&envelope),
            format!(
                "bw 01: 2 of 5 matching lines — the newest; the 3 earlier are left out (narrow the \
                 search to see them)\n2026 WARN x\n{cut}…\n\nww 02: no answer (504)\n\n\
                 aw 03: 1 of 1 matching lines\n2026 ERROR y\n\n\
                 lw 04: 1 matching lines — the newest; earlier ones may be left out (narrow the \
                 search, or name `since`, to see them)\n2026 INFO z"
            )
        );
        assert_eq!(render(&json!({"members": []})), "No member answered.");
    }
}
