//! `keep_state {host}`: how a machine's keep stands
//! (`docs/claude_zend_gateway.md` §8.4) — each service it keeps: its stable
//! file's build, whether it runs, the trial running on it and how the last one
//! ended. An Admin's.

use std::time::Duration;

use axum::http::Method;
use serde_json::{json, Value};

use super::cluster_status::ago;
use super::{string, Answer, Ctx};
use crate::mcp::ops::{self, json_of};
use crate::mcp::workers::now_ms;

pub const NAME: &str = "keep_state";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Read a machine's keep: each service it keeps (motherd, web) — the build its stable \
            file holds, whether it runs, any trial running on it, and how the last trial ended, proven or \
            rolled back. Read it after `keep_trial` to learn the trial's outcome. An Admin's.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "The machine, as `cluster_status` names it."}
            },
            "required": ["host"],
            "additionalProperties": false
        }
    })
}

/// A keep's services, one block each.
pub fn render(host: &str, state: &Value, now: u64) -> String {
    let mut out = format!("keep on {host}:");
    for s in state
        .get("services")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let name = s.get("name").and_then(Value::as_str).unwrap_or("?");
        let build = match s.get("build").filter(|b| !b.is_null()) {
            Some(b) => {
                let local = b.get("local").and_then(Value::as_str).unwrap_or("");
                format!(
                    "{}-{}{}",
                    b.get("count").map_or_else(|| "?".into(), Value::to_string),
                    b.get("hash").and_then(Value::as_str).unwrap_or("?"),
                    if local.is_empty() {
                        String::new()
                    } else {
                        format!(" (local {local})")
                    }
                )
            }
            None => "unreadable".into(),
        };
        let running = match s.get("running").and_then(Value::as_u64) {
            Some(pid) => format!("running as pid {pid}"),
            None => "not running".into(),
        };
        out.push_str(&format!("\n- {name}: build {build}, {running}"));
        let trials = s.get("trials").cloned().unwrap_or(Value::Null);
        if let Some(t) = trials.get("running").filter(|t| !t.is_null()) {
            out.push_str(&format!(
                "\n  trial of {} running, started {}",
                t.get("artifact").and_then(Value::as_str).unwrap_or("?"),
                ago(
                    t.get("started_ms").and_then(Value::as_u64).unwrap_or(0),
                    now
                )
            ));
        }
        if let Some(t) = trials.get("last").filter(|t| !t.is_null()) {
            out.push_str(&format!(
                "\n  last trial, of {}: {} {} — {}",
                t.get("artifact").and_then(Value::as_str).unwrap_or("?"),
                if t.get("proven").and_then(Value::as_bool) == Some(true) {
                    "proven"
                } else {
                    "rolled back"
                },
                ago(t.get("ended_ms").and_then(Value::as_u64).unwrap_or(0), now),
                t.get("detail")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .replace('\n', " / ")
            ));
        }
    }
    out
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    let host = match string(args, "host") {
        Ok(h) => h,
        Err(e) => return Answer::error(e),
    };
    let reply = ops::to_keep(
        &ctx.session,
        &ctx.caller,
        Method::GET,
        "/v1/keep/state",
        host,
        None,
        Duration::from_secs(20),
    )
    .await;
    match reply.and_then(|r| json_of(&r)) {
        Ok(v) => Answer::ok(render(host, &v, now_ms())),
        Err(e) => Answer::error(format!("keep on {host}: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_keep_reads_as_its_services_and_their_trials() {
        let now = 1_800_000_000_000;
        let state = json!({"services": [
            {"name": "motherd", "running": 4242,
             "build": {"count": 5739, "hash": "c544", "local": ""},
             "trials": {"running": null,
                        "last": {"artifact": "stage/motherd.exe", "proven": true,
                                 "detail": "proven", "ended_ms": now - 120_000}}},
            {"name": "web", "running": null, "build": null,
             "trials": {"running": {"artifact": "stage/web.exe", "started_ms": now - 30_000},
                        "last": null}}
        ]});
        assert_eq!(
            render("blackworld", &state, now),
            "keep on blackworld:\n- motherd: build 5739-c544, running as pid 4242\n  last trial, \
             of stage/motherd.exe: proven 2 min ago — proven\n- web: build unreadable, not \
             running\n  trial of stage/web.exe running, started 30 s ago"
        );
    }
}
