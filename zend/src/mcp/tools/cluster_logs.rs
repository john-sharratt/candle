//! `cluster_logs`: what the members' logs say (`docs/claude_zend_gateway.md`
//! §8.1) — `GET /v1/logs`, which the gateway asks every member and gathers;
//! with `node`, one member's, which is for Admins only.
//!
//! **A machine's log is its daemon's, not one app's.** The gateway gathers
//! `/v1/logs` from zend's members, so a machine whose zend is still loading —
//! for ten minutes after every restart — answered nothing, though its npcd
//! member was up and writing to the same log. For an Admin, a machine the
//! gathering missed is read through another of its members, by the gateway's
//! member route (`/gateway/member/<app>/<node>/v1/logs`), and a `node` names a
//! member of any app.

use std::collections::HashSet;
use std::time::Duration;

use axum::http::Method;
use futures::future::join_all;
use serde_json::{json, Value};
use url::form_urlencoded::Serializer;

use super::{Answer, Ctx};
use crate::mcp::ops::{self, json_of};
use crate::mcp::ops_tools::OpsCall;

pub const NAME: &str = "cluster_logs";

/// The apps whose members serve `/v1/logs`: the daemon's tenants.
const LOGGING_APPS: [&str; 2] = ["zend", "npcd"];

/// The app the gateway gathers `/v1/logs` from for this tool.
const GATHERED_APP: &str = "zend";

/// How long one member has to answer a search.
const WITHIN: Duration = Duration::from_secs(30);

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

/// One member as the gateway's status lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Listed {
    pub app: String,
    pub host: String,
    pub node: String,
}

/// The members of the logging apps in the gateway's `status`, a ready one
/// before the others of its machine.
pub fn listed(status: &Value) -> Vec<Listed> {
    let mut members: Vec<(bool, Listed)> = status
        .get("members")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|m| {
            let field = |k: &str| m.get(k).and_then(Value::as_str).map(str::to_string);
            let app = field("app")?;
            LOGGING_APPS.contains(&app.as_str()).then_some(())?;
            let ready = m.get("state").and_then(Value::as_str) == Some("ready");
            Some((
                !ready,
                Listed {
                    app,
                    host: field("host")?,
                    node: field("node")?,
                },
            ))
        })
        .collect();
    members.sort_by_key(|(not_ready, _)| *not_ready);
    members.into_iter().map(|(_, m)| m).collect()
}

/// For each machine `envelope` holds no answer from, one of its members in
/// `members` to read it through.
pub fn missed(envelope: &Value, members: &[Listed]) -> Vec<Listed> {
    let mut answered: HashSet<String> = envelope
        .get("members")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|m| m.get("status").and_then(Value::as_u64) == Some(200))
        .filter_map(|m| m.get("host").and_then(Value::as_str).map(str::to_string))
        .collect();
    members
        .iter()
        .filter(|m| answered.insert(m.host.clone()))
        .cloned()
        .collect()
}

/// `envelope` with each machine it missed answered by `through`: a machine
/// that came back in place of its unanswered entry, the rest after.
pub fn with_answers(mut envelope: Value, through: Vec<Value>) -> Value {
    let members = envelope
        .get_mut("members")
        .and_then(Value::as_array_mut)
        .map(std::mem::take)
        .unwrap_or_default();
    let answered: HashSet<String> = through
        .iter()
        .filter_map(|m| m.get("host").and_then(Value::as_str).map(str::to_string))
        .collect();
    let mut out: Vec<Value> = members
        .into_iter()
        .filter(|m| {
            m.get("host")
                .and_then(Value::as_str)
                .is_none_or(|h| !answered.contains(h))
        })
        .collect();
    out.extend(through);
    envelope["members"] = Value::Array(out);
    envelope
}

/// The gateway's route to one member's search.
fn member_path(member: &Listed, search: &str) -> String {
    format!("/gateway/member/{}/{}{search}", member.app, member.node)
}

/// `member`'s search, as an envelope entry.
async fn read_through(call: &OpsCall, member: &Listed, search: &str) -> Value {
    let reply = call
        .up(Method::GET, &member_path(member, search), None, WITHIN)
        .await;
    let (status, body) = match reply {
        Ok(r) => (
            r.status,
            serde_json::from_slice::<Value>(&r.body).unwrap_or(Value::Null),
        ),
        Err(_) => (502, Value::Null),
    };
    json!({
        "host": member.host,
        "node": member.node,
        "through": member.app,
        "status": status,
        "body": body,
    })
}

/// The gateway's status, for an Admin; `None` for anyone else or when it
/// cannot be read.
async fn gateway_members(call: &OpsCall) -> Option<Vec<Listed>> {
    if !call.admin() {
        return None;
    }
    let reply = call
        .up(Method::GET, "/gateway/status.json?log=0", None, WITHIN)
        .await
        .ok()?;
    json_of(&reply).ok().map(|s| listed(&s))
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
        let node = match m.get("through").and_then(Value::as_str) {
            Some(app) => format!(
                "{} (its {app} member)",
                m.get("node").and_then(Value::as_str).unwrap_or("?")
            ),
            None => m
                .get("node")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string(),
        };
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

pub async fn run(call: &OpsCall, args: &Value) -> Answer {
    let search = path_of(args);
    if let Some(n) = args.get("node").and_then(Value::as_str) {
        if !call.admin() {
            return Answer::error("`cluster_logs` with `node` is an Admin's");
        }
        let node = match ops::node(n) {
            Ok(n) => n,
            Err(e) => return Answer::error(e),
        };
        return one(call, &node, &search).await;
    }
    let (reply, members) = tokio::join!(
        call.up(Method::GET, &search, None, WITHIN),
        gateway_members(call)
    );
    let envelope = match reply.and_then(|r| json_of(&r)) {
        Ok(v) => v,
        Err(e) => return Answer::error(format!("/v1/logs: {e}")),
    };
    let Some(members) = members else {
        return Answer::ok(render(&envelope));
    };
    let missed = missed(&envelope, &members);
    let through = join_all(missed.iter().map(|m| read_through(call, m, &search))).await;
    Answer::ok(render(&with_answers(envelope, through)))
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    run(&OpsCall::of(ctx), args).await
}

/// Member `node`'s search: a zend member's as the gateway routes it, any
/// other logging app's through the gateway's member route.
async fn one(call: &OpsCall, node: &str, search: &str) -> Answer {
    let listed = gateway_members(call).await.unwrap_or_default();
    let member = listed.iter().find(|m| m.node == node);
    let entry = match member {
        Some(m) if m.app != GATHERED_APP => read_through(call, m, search).await,
        _ => {
            let reply = match call.up(Method::GET, search, Some(node), WITHIN).await {
                Ok(r) => r,
                Err(e) => return Answer::error(e),
            };
            match json_of(&reply) {
                Ok(v) => json!({
                    "host": member.map_or("", |m| m.host.as_str()),
                    "node": node,
                    "status": 200,
                    "body": v,
                }),
                Err(e) => return Answer::error(format!("/v1/logs: {e}")),
            }
        }
    };
    Answer::ok(render(&json!({ "members": [entry] })))
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

    fn status() -> Value {
        json!({"members": [
            {"app": "zend", "host": "BlackWorld", "node": "00000000000000b1", "state": "ready"},
            {"app": "agent", "host": "WhiteWorld", "node": "00000000000000a6", "state": "ready"},
            {"app": "zend", "host": "WhiteWorld", "node": "00000000000000c6", "state": "loading"},
            {"app": "npcd", "host": "WhiteWorld", "node": "00000000000000d6", "state": "ready"},
            {"app": "keep", "host": "whiteworld", "node": "00000000000000e6", "state": "ready"}
        ]})
    }

    /// **A machine whose zend did not answer is read through its other
    /// member**: only the daemon's tenants are listed, a ready one first, and
    /// a machine that answered is not asked again.
    #[test]
    fn a_machine_the_gathering_missed_is_read_through_another_member() {
        let members = listed(&status());
        assert_eq!(
            members
                .iter()
                .map(|m| (m.app.as_str(), m.node.as_str()))
                .collect::<Vec<_>>(),
            [
                ("zend", "00000000000000b1"),
                ("npcd", "00000000000000d6"),
                ("zend", "00000000000000c6"),
            ],
            "agents and keeps hold no daemon log; ready members first"
        );
        let envelope = json!({"members": [
            {"host": "BlackWorld", "node": "00000000000000b1", "status": 200, "body": {"lines": []}},
        ]});
        let missed = missed(&envelope, &members);
        assert_eq!(
            missed,
            [Listed {
                app: "npcd".into(),
                host: "WhiteWorld".into(),
                node: "00000000000000d6".into(),
            }],
            "WhiteWorld, through its ready npcd member"
        );
        assert_eq!(
            member_path(&missed[0], "/v1/logs?limit=5"),
            "/gateway/member/npcd/00000000000000d6/v1/logs?limit=5"
        );
    }

    /// **The machine read through takes the place of its unanswered entry**,
    /// and says which member answered for it.
    #[test]
    fn the_machine_read_through_replaces_its_unanswered_entry() {
        let envelope = json!({"members": [
            {"host": "BlackWorld", "node": "01", "status": 200,
             "body": {"lines": ["2026 INFO a"], "matched": 1}},
            {"host": "WhiteWorld", "node": "02", "status": 504},
        ]});
        let through = vec![json!({
            "host": "WhiteWorld", "node": "03", "through": "npcd", "status": 200,
            "body": {"lines": ["2026 INFO npcd::discord: b"], "matched": 1},
        })];
        assert_eq!(
            render(&with_answers(envelope, through)),
            "BlackWorld 01: 1 of 1 matching lines\n2026 INFO a\n\n\
             WhiteWorld 03 (its npcd member): 1 of 1 matching lines\n2026 INFO npcd::discord: b"
        );
    }
}
