//! `service_status {host?}`: what each machine's keep runs (`docs/jobs.md`):
//! each service's installed build, whether it runs, the trial running and how
//! the last one ended, with the deploy each belongs to, and the deploys still
//! building there. A deploy itself is a job, watched with `job_status`.

use agent::client::Agents;
use axum::http::Method;
use serde_json::{json, Value};

use super::{built_service, keep, keep_build, KEEP_STATE, PROOF_WINDOW_MS};
use crate::mcp::ops;
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::tools::cluster_status::ago;
use crate::mcp::tools::job::JobHandle;
use crate::mcp::tools::repo_job::{phase, place, span, Handle, NO_AGENTS};
use crate::mcp::tools::{string, Answer, Ctx};
use crate::mcp::workers::now_ms;

pub const NAME: &str = "service_status";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "Say what each machine's keep runs — or `host`'s alone: each service's installed build, \
            whether it runs, the trial keep is running and how the last one ended, with the deploy each \
            belongs to, and the deploys still building there. Read it before a deploy, to see what each \
            machine runs and that nothing else is deploying there. A deploy itself is a job: wait on it with \
            `job_status`, read it with `job_read`, stop it with `job_cancel`. Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "minLength": 1, "maxLength": 32, "description": "One machine only, by name; every machine when left out."}
            },
            "additionalProperties": false
        }
    })
}

/// What `trial`, a trial on keep's board, says of how it ended.
fn ended(trial: &Value, now: u64) -> String {
    let when = ago(
        trial.get("ended_ms").and_then(Value::as_u64).unwrap_or(0),
        now,
    );
    let detail = trial
        .get("detail")
        .and_then(Value::as_str)
        .unwrap_or("")
        .replace('\n', " / ");
    if trial.get("proven").and_then(Value::as_bool) == Some(true) {
        format!("proven {when}")
    } else if trial.get("cancelled").and_then(Value::as_bool) == Some(true) {
        format!("cancelled and rolled back {when} — {detail}")
    } else {
        format!("rolled back {when} — {detail}")
    }
}

/// How far into its proof window `trial`, running on keep's board, is.
pub fn trying(trial: &Value, now: u64) -> String {
    let started = trial
        .get("started_ms")
        .and_then(Value::as_u64)
        .unwrap_or(now);
    format!(
        "keep is trying it: {} in, of a {}-minute window to print that it serves — the window \
         restarts while it is catching up on replication{}",
        span(now.saturating_sub(started)),
        PROOF_WINDOW_MS / 60_000,
        if trial.get("cancelled").and_then(Value::as_bool) == Some(true) {
            "; cancelled, rolling back"
        } else {
            ""
        }
    )
}

/// A keep's services, one block each, naming the deploy each trial belongs to.
pub fn render_keep(host: &str, state: &Value, now: u64) -> String {
    let mut out = match keep_build(host, state) {
        Ok(build) => format!("keep on {host} (keep {build}):"),
        Err(why) => format!("keep on {host}: {why}"),
    };
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
        let of = |t: &Value| {
            let artifact = t.get("artifact").and_then(Value::as_str).unwrap_or("?");
            match t.get("deploy").and_then(Value::as_str) {
                Some(d) => format!("deploy:{d} ({artifact})"),
                None => artifact.to_string(),
            }
        };
        let trials = s.get("trials").cloned().unwrap_or(Value::Null);
        if let Some(t) = trials.get("running").filter(|t| !t.is_null()) {
            out.push_str(&format!("\n  trial of {} — {}", of(t), trying(t, now)));
        }
        if let Some(t) = trials.get("last").filter(|t| !t.is_null()) {
            out.push_str(&format!("\n  last trial, of {}: {}", of(t), ended(t, now)));
        }
    }
    out
}

/// The deploys building on agent `node`, a line each.
async fn building_on(agents: &Agents, node: u64, now: u64) -> Vec<String> {
    let Ok(jobs) = agents.jobs_of(agents.at(node), 40).await else {
        return Vec::new();
    };
    jobs.jobs
        .iter()
        .filter(|j| !j.status.is_finished())
        .filter_map(|j| {
            let service = built_service(&j.command)?;
            Some(format!(
                "  {} of {service} building: {}, for {}",
                JobHandle::Deploy(Handle::new(node, &j.id)),
                phase(&j.phase, node),
                span(now.saturating_sub(j.started_ms))
            ))
        })
        .collect()
}

pub async fn run(call: &OpsCall, args: &Value) -> Answer {
    let Some(agents) = call.session.agents() else {
        return Answer::error(NO_AGENTS);
    };
    let only = string(args, "host").ok();
    let listed = match agents.listing().await {
        Ok(l) => l,
        Err(e) => return Answer::error(format!("the agents could not be listed: {e}")),
    };
    let mut hosts: Vec<(String, u64)> = Vec::new();
    for a in &listed {
        let Ok(node) = u64::from_str_radix(&a.node, 16) else {
            continue;
        };
        if only.is_some_and(|o| !o.eq_ignore_ascii_case(&a.host)) {
            continue;
        }
        if !hosts.iter().any(|(h, _)| h.eq_ignore_ascii_case(&a.host)) {
            hosts.push((a.host.clone(), node));
        }
    }
    if hosts.is_empty() {
        return Answer::error(match only {
            Some(h) => {
                format!("no agent is connected on {h}, so its keep is not reached through one")
            }
            None => "no agent is connected".to_string(),
        });
    }
    let now = now_ms();
    let mut blocks = Vec::new();
    for (host, node) in hosts {
        let mut block = match keep(call, Method::GET, KEEP_STATE, &host, None).await {
            Ok(state) => render_keep(&host, &state, now),
            Err(e) => e,
        };
        let building = building_on(&agents, node, now).await;
        if !building.is_empty() {
            block.push_str(&format!(
                "\n- deploys building on its agent ({}):\n",
                place(node)
            ));
            block.push_str(&building.join("\n"));
        }
        blocks.push(block);
    }
    Answer::ok(blocks.join("\n"))
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    run(&OpsCall::of(ctx), args).await
}

#[cfg(test)]
mod tests {
    use super::super::REBUILD_KEEP;
    use super::*;

    const NOW: u64 = 1_800_000_000_000;

    /// **A keep's services name the deploy each trial belongs to**, by its
    /// job handle, and a cancelled one says so.
    #[test]
    fn a_keep_names_the_deploy_of_each_trial() {
        let state = json!({"keep": {"count": 4543, "hash": "1a2b3c4d5", "local": ""}, "services": [
            {"name": "motherd", "running": 4242,
             "build": {"count": 5739, "hash": "c544", "local": ""},
             "trials": {"running": null,
                        "last": {"artifact": "stage/motherd.exe", "deploy": "00000000000000ab:Xe",
                                 "proven": false, "cancelled": true,
                                 "detail": "rolled back\nthe trial was cancelled", "ended_ms": NOW - 120_000}}},
            {"name": "web", "running": null, "build": null,
             "trials": {"running": {"artifact": "stage/web.exe", "deploy": null, "started_ms": NOW - 30_000},
                        "last": null}}
        ]});
        let text = render_keep("BlackWorld", &state, NOW);
        assert!(text.starts_with("keep on BlackWorld (keep 4543-1a2b3c4d5):\n- motherd: build 5739-c544, running as pid 4242\n  last trial, of deploy:00000000000000ab:Xe (stage/motherd.exe): cancelled and rolled back 2 min ago"), "{text}");
        assert!(text.contains("\n- web: build unreadable, not running\n  trial of stage/web.exe — keep is trying it: 30 s in"), "{text}");
    }

    /// **A keep too old to record deploys says so at the top**, with how to
    /// update it.
    #[test]
    fn an_old_keep_is_named_with_its_fix() {
        let text = render_keep("WhiteWorld", &json!({"services": []}), NOW);
        assert_eq!(
            text,
            format!(
                "keep on WhiteWorld: keep on WhiteWorld is too old to record which deploy a trial \
                 belongs to, so a deploy there would build and try but never see its trial; update \
                 keep first: {REBUILD_KEEP}"
            )
        );
    }
}
