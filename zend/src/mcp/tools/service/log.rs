//! `service_log {host, service, mode, …}`: read a service's trial log on a
//! machine, where keep keeps it (`docs/build_deploy_tools.md` §3) — the
//! service's stderr since its last trial began, which a proven trial goes on
//! writing as the service. The same four modes a job's log is read in.

use axum::http::Method;
use cluster::log_read::LogRead;
use serde_json::{json, Value};

use super::{keep, service_of, KEEP_LOG, SERVICES};
use crate::mcp::ops;
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::tools::repo_job::log::{ask_of, log_properties, log_rules, render};
use crate::mcp::tools::{string, Answer, Ctx};

pub const NAME: &str = "service_log";

pub fn definition() -> Value {
    let mut properties = json!({
        "host": {"type": "string", "minLength": 1, "maxLength": 32, "description": "The machine, by name: `BlackWorld`."},
        "service": {"type": "string", "enum": SERVICES, "description": "`motherd`, `web` or `mcp`."}
    });
    for (k, v) in log_properties().as_object().expect("an object") {
        properties[k] = v.clone();
    }
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "Read a service's trial log on one machine, where keep, the supervisor that runs the \
            machine's services, writes it: the service's stderr from the moment its last trial started — \
            loading, catching up on replication, `INFO serving` when it proved itself, or the panic or exit \
            that rolled it back — and, after a proven trial, everything the service has printed since. Read \
            it on keep's machine so only the answer comes back, every line numbered as in the log. Start \
            with `mode: summary` (errors, panics, the last lines), read around a numbered line with `mode: \
            lines`, search with `mode: grep`, or follow a running trial with `mode: tail`. A deploy's build \
            log is not here: read it with `repo_job_log {job: <deploy>}`. Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": ["host", "service", "mode"],
            "allOf": log_rules(),
            "additionalProperties": false
        }
    })
}

pub async fn run(call: &OpsCall, args: &Value) -> Answer {
    let (host, service) = match (
        string(args, "host"),
        string(args, "service").and_then(service_of),
    ) {
        (Ok(h), Ok(s)) => (h, s),
        (Err(e), _) | (_, Err(e)) => return Answer::error(e),
    };
    let ask = match ask_of(args) {
        Ok(a) => a,
        Err(e) => return Answer::error(e),
    };
    let body = json!({"service": service, "ask": ask});
    let read: LogRead = match keep(call, Method::POST, KEEP_LOG, host, Some(&body))
        .await
        .and_then(|v| serde_json::from_value(v).map_err(|e| format!("keep on {host}: {e}")))
    {
        Ok(r) => r,
        Err(e) => return Answer::error(e),
    };
    let what = format!(
        "{service}'s trial log on {host} ({})",
        if read.ended {
            "no trial running"
        } else {
            "a trial is running"
        }
    );
    let reread = |rest: String| {
        format!("`service_log {{host: \"{host}\", service: \"{service}\", {rest}}}`")
    };
    Answer::ok(render(&what, &ask, &read, &reread))
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    run(&OpsCall::of(ctx), args).await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The machine and service come before the mode, and the mode before
    /// what it needs.**
    #[test]
    fn the_schema_names_the_log_then_how_to_read_it() {
        let d = definition();
        let props: Vec<&String> = d["inputSchema"]["properties"]
            .as_object()
            .unwrap()
            .keys()
            .collect();
        assert_eq!(props[..3], ["host", "service", "mode"]);
        assert_eq!(
            d["inputSchema"]["required"],
            json!(["host", "service", "mode"])
        );
    }
}
