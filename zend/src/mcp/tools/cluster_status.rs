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
        "annotations": {"readOnlyHint": true},
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

/// The build under `key`, as the gateway and every member write one —
/// `{count, hash, local}` — read as `count-hash`, with ` (local)` for a
/// build with uncommitted changes; `None` when there is none.
fn build_of(v: &Value, key: &str) -> Option<String> {
    let b = v.get(key).filter(|b| b.is_object())?;
    let local = b
        .get("local")
        .and_then(Value::as_str)
        .is_some_and(|l| !l.is_empty());
    Some(format!(
        "{}-{}{}",
        b.get("count").and_then(Value::as_u64)?,
        b.get("hash").and_then(Value::as_str)?,
        if local { " (local)" } else { "" }
    ))
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
            build_of(&body, "build").as_deref().unwrap_or("?"),
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
        build_of(status, "build").as_deref().unwrap_or("?"),
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
            "\n- {}: {}, {} of {} ready",
            str_of(app, "app"),
            str_of(app, "mode"),
            app.get("ready").and_then(Value::as_u64).unwrap_or(0),
            app.get("members").and_then(Value::as_u64).unwrap_or(0),
        ));
        // A service is admitted on any build and never rolled out: it has
        // no floor to name.
        if app.get("service").and_then(Value::as_bool) == Some(true) {
            out.push_str(", a service (no floor, released per machine)");
            continue;
        }
        out.push_str(&format!(
            ", floor {}",
            build_of(app, "floor").as_deref().unwrap_or("none")
        ));
        if let Some(p) = build_of(app, "pending_floor") {
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
            "\n  {} {} {} build {}: {}",
            str_of(m, "app"),
            str_of(m, "host"),
            str_of(m, "node"),
            build_of(m, "build").as_deref().unwrap_or("?"),
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
    let members = ops::up(
        &ctx.session,
        &ctx.caller,
        Method::GET,
        "/v1/status",
        None,
        None,
        within,
    );
    // The gateway's own view is asked for beside the members', not after it.
    let gateway = async {
        if !ctx.caller.is_admin() {
            return None;
        }
        Some(
            ops::up(
                &ctx.session,
                &ctx.caller,
                Method::GET,
                "/gateway/status.json?log=0",
                None,
                None,
                within,
            )
            .await
            .and_then(|r| json_of(&r)),
        )
    };
    let (reply, gateway) = tokio::join!(members, gateway);
    let reply = match reply {
        Ok(r) => r,
        Err(e) => return Answer::error(e),
    };
    let mut out = match json_of(&reply) {
        Ok(v) => render_members(&v, now_ms()),
        Err(e) => return Answer::error(format!("/v1/status: {e}")),
    };
    if let Some(gateway) = gateway {
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
                    "state": "ready", "build": {"count": 5739, "hash": "c544", "local": ""}, "detail": "",
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

    /// The gateway's status as `/gateway/status.json` writes it: every build
    /// an object, and a candidate's already a name.
    #[test]
    fn the_gateways_view_names_apps_members_and_candidates() {
        let b = |count: u64, hash: &str, local: &str| json!({"count": count, "hash": hash, "local": local});
        let status = json!({
            "gateway": "bw", "build": b(1, "a", ""), "draining": false,
            "apps": [
                {"app": "zend", "service": false, "mode": "clustered", "ready": 2, "members": 2,
                 "floor": b(10, "h", ""), "pending_floor": b(11, "h", "")},
                {"app": "npcd", "service": false, "mode": "dumb", "ready": 0, "members": 0,
                 "floor": null, "pending_floor": null},
                {"app": "mcp", "service": true, "mode": "clustered", "ready": 2, "members": 2,
                 "floor": null, "pending_floor": null}
            ],
            "members": [
                {"app": "zend", "host": "bw", "node": "01", "build": b(10, "h", ""), "local_build": false, "state": "ready"},
                {"app": "mcp", "host": "ww", "node": "03", "build": b(12, "j", "9f"), "local_build": true, "state": "ready"}
            ],
            "upgrades": [{"app": "zend", "candidate": {"node": "02", "build": "11-h", "state": "certifying"}}],
            "faults": []
        });
        assert_eq!(
            render_gateway(&status),
            "gateway bw (build 1-a)\n\
             - zend: clustered, 2 of 2 ready, floor 10-h, pending floor 11-h\n\
             - npcd: dumb, 0 of 0 ready, floor none\n\
             - mcp: clustered, 2 of 2 ready, a service (no floor, released per machine)\n  \
             zend bw 01 build 10-h: ready\n  \
             mcp ww 03 build 12-j (local): ready\n\
             upgrade zend: candidate 11-h on 02, \"certifying\""
        );
    }

    #[test]
    fn a_time_reads_as_how_long_ago() {
        assert_eq!(ago(NOW - 30_000, NOW), "30 s ago");
        assert_eq!(ago(NOW - 600_000, NOW), "10 min ago");
        assert_eq!(ago(NOW - 36_000_000, NOW), "10.0 h ago");
    }
}
