//! `cluster_status`: how the cluster stands (`docs/claude_zend_gateway.md`
//! §8.1) — every zend member's own `/v1/status`, gathered by the gateway, and
//! for an Admin the gateway's view: apps, members, upgrades, faults.

use std::time::Duration;

use axum::http::Method;
use serde_json::{json, Value};

use super::{Answer, Ctx};
use crate::mcp::ops::{self, json_of};
use crate::mcp::workers::now_ms;

pub const NAME: &str = "cluster_status";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "How the cluster stands: each member — a machine running the motherd daemon — with \
            its machine name, node id, build, state, when it started, and how each app it hosts stands; and, \
            for an Admin, the gateway's view: each app's floor (the oldest \
            build it still routes to) and ready members, upgrades in progress, members in fault. Check it \
            before delegating heavy work, before and after any operation, and whenever a task stalls.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }
    })
}

/// How long ago `ms` was, as a person reads it.
pub fn ago(ms: u64, now: u64) -> String {
    let s = now.saturating_sub(ms) / 1000;
    match s {
        0..=89 => format!("{s} s ago"),
        90..=5_399 => format!("{} min ago", s / 60),
        _ => format!("{:.1} h ago", s as f64 / 3600.0),
    }
}

fn str_of<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or("?")
}

/// The members' own statuses, as the gateway's envelope gathered them.
pub fn render_members(envelope: &Value, now: u64) -> String {
    let members = envelope
        .get("members")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut out = format!("Members answering /v1/status: {}", members.len());
    for m in &members {
        let (host, node) = (str_of(m, "host"), str_of(m, "node"));
        let code = m.get("status").and_then(Value::as_u64).unwrap_or(0);
        let body = m.get("body").cloned().unwrap_or(Value::Null);
        if code != 200 {
            out.push_str(&format!(
                "\n- {host} {node}: no answer ({code} {})",
                m.get("error").and_then(Value::as_str).unwrap_or("")
            ));
            continue;
        }
        let mut line = format!(
            "\n- {host} {node} build {}: {}",
            str_of(&body, "build"),
            str_of(&body, "state")
        );
        if let Some(d) = body
            .get("detail")
            .and_then(Value::as_str)
            .filter(|d| !d.is_empty())
        {
            line.push_str(&format!(" — {d}"));
        }
        if let Some(at) = body.get("started_at_ms").and_then(Value::as_u64) {
            line.push_str(&format!(", started {}", ago(at, now)));
        }
        if let Some(tenants) = body.get("tenants").and_then(Value::as_object) {
            let t: Vec<String> = tenants
                .iter()
                .map(|(name, s)| format!("{name} {}", str_of(s, "state")))
                .collect();
            line.push_str(&format!("; tenants: {}", t.join(", ")));
        }
        out.push_str(&line);
    }
    if let Some(group) = envelope.get("group").filter(|g| !g.is_null()) {
        out.push_str(&format!("\ngroup: {group}"));
    }
    out
}

/// The gateway's own view, for an Admin.
pub fn render_gateway(status: &Value) -> String {
    let mut out = format!(
        "gateway {} (build {}){}",
        str_of(status, "gateway"),
        str_of(status, "build"),
        if status.get("draining").and_then(Value::as_bool) == Some(true) {
            ", draining"
        } else {
            ""
        }
    );
    for app in status
        .get("apps")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        out.push_str(&format!(
            "\n- {}: {}, {} of {} ready, floor {}",
            str_of(app, "app"),
            str_of(app, "mode"),
            app.get("ready").and_then(Value::as_u64).unwrap_or(0),
            app.get("members").and_then(Value::as_u64).unwrap_or(0),
            app.get("floor").and_then(Value::as_str).unwrap_or("none"),
        ));
        if let Some(p) = app.get("pending_floor").and_then(Value::as_str) {
            out.push_str(&format!(", pending floor {p}"));
        }
    }
    for m in status
        .get("members")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        out.push_str(&format!(
            "\n  {} {} {} build {}{}: {}",
            str_of(m, "app"),
            str_of(m, "host"),
            str_of(m, "node"),
            str_of(m, "build"),
            if m.get("local_build").and_then(Value::as_bool) == Some(true) {
                " (local)"
            } else {
                ""
            },
            str_of(m, "state"),
        ));
    }
    for u in status
        .get("upgrades")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(c) = u.get("candidate").filter(|c| !c.is_null()) {
            out.push_str(&format!(
                "\nupgrade {}: candidate {} on {}, {}",
                str_of(u, "app"),
                str_of(c, "build"),
                str_of(c, "node"),
                c.get("state").map_or_else(|| "?".into(), Value::to_string),
            ));
        }
    }
    let faults = status
        .get("faults")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    if faults > 0 {
        out.push_str(&format!("\nmembers in or near fault: {faults}"));
    }
    out
}

pub async fn call(ctx: &Ctx, _args: &Value) -> Answer {
    let within = Duration::from_secs(15);
    let reply = match ops::up(
        &ctx.session,
        &ctx.caller,
        Method::GET,
        "/v1/status",
        None,
        None,
        within,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => return Answer::error(e),
    };
    let mut out = match json_of(&reply) {
        Ok(v) => render_members(&v, now_ms()),
        Err(e) => return Answer::error(format!("/v1/status: {e}")),
    };
    if ctx.caller.is_admin() {
        let gateway = ops::up(
            &ctx.session,
            &ctx.caller,
            Method::GET,
            "/gateway/status.json?log=0",
            None,
            None,
            within,
        )
        .await
        .and_then(|r| json_of(&r));
        out.push_str("\n\n");
        match gateway {
            Ok(v) => out.push_str(&render_gateway(&v)),
            Err(e) => out.push_str(&format!("the gateway's own view could not be read: {e}")),
        }
    }
    Answer::ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_800_000_000_000;

    #[test]
    fn a_members_line_says_its_state_its_start_and_its_tenants() {
        let envelope = json!({
            "members": [
                {"node": "00000000000000ab", "host": "blackworld", "status": 200, "body": {
                    "state": "ready", "build": "5739-c544", "detail": "",
                    "started_at_ms": NOW - 7_200_000,
                    "tenants": {"npcd": {"state": "ready"}, "zend": {"state": "loading"}}
                }},
                {"node": "00000000000000cd", "host": "whiteworld", "status": 504, "error": "deadline"}
            ],
            "group": {"members": 2}
        });
        assert_eq!(
            render_members(&envelope, NOW),
            "Members answering /v1/status: 2\n\
             - blackworld 00000000000000ab build 5739-c544: ready, started 2.0 h ago; tenants: \
             npcd ready, zend loading\n\
             - whiteworld 00000000000000cd: no answer (504 deadline)\n\
             group: {\"members\":2}"
        );
    }

    #[test]
    fn the_gateways_view_names_apps_members_and_candidates() {
        let status = json!({
            "gateway": "bw", "build": "1-a", "draining": false,
            "apps": [{"app": "zend", "mode": "clustered", "ready": 2, "members": 2, "floor": "10-h"}],
            "members": [{"app": "zend", "host": "bw", "node": "01", "build": "10-h", "local_build": false, "state": "ready"}],
            "upgrades": [{"app": "zend", "candidate": {"node": "02", "build": "11-h", "state": "certifying"}}],
            "faults": []
        });
        assert_eq!(
            render_gateway(&status),
            "gateway bw (build 1-a)\n- zend: clustered, 2 of 2 ready, floor 10-h\n  \
             zend bw 01 build 10-h: ready\nupgrade zend: candidate 11-h on 02, \"certifying\""
        );
    }

    #[test]
    fn a_time_reads_as_how_long_ago() {
        assert_eq!(ago(NOW - 30_000, NOW), "30 s ago");
        assert_eq!(ago(NOW - 600_000, NOW), "10 min ago");
        assert_eq!(ago(NOW - 36_000_000, NOW), "10.0 h ago");
    }
}
