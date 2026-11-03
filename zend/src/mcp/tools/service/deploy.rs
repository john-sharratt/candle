//! `service_deploy {host, service}`: build a service from origin/main on one
//! machine's agent and try it there through keep (`docs/build_deploy_tools.md`
//! §4). It answers once the build job is accepted, with the deploy's handle;
//! a driver task in this member waits for the build and asks keep for the
//! trial, tagged with the handle, and keep runs the trial from there.

use std::time::Duration;

use agent::client::Agents;
use agent::wire::{RunJob, Staged};
use axum::http::Method;
use serde_json::{json, Value};
use zend_vfs::{JobStatus, Snapshot};

use super::{guard, keep, service_of, trial_body, KEEP_TRIAL, SERVICES};
use crate::mcp::ops;
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::tools::repo_job::{on_host, phase, Handle, NO_AGENTS};
use crate::mcp::tools::{string, Answer, Ctx};

pub const NAME: &str = "service_deploy";

/// The repository every service is built from, and the one branch keep tries.
const REPO: &str = "candle";
const BRANCH: &str = "main";

/// How long a build may run: a first one on a machine compiles the workspace.
const BUILD_TIMEOUT_S: u64 = 3_600;

/// The longest one wait on the build is held.
const STEP: Duration = Duration::from_secs(25);

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Deploy a service to one machine: build it as origin/main holds it on that machine's \
            build agent (`cargo build --release -p <service>`), stage the binary, and try it through keep, \
            the supervisor that runs the machine's services. keep starts the new build in place of the \
            installed one and keeps it once it prints that it is serving — within ten minutes, the window \
            restarting while it catches up on replication — and otherwise puts the previous binary back, \
            so a bad build cannot keep a machine down. Answers at once with the deploy's handle \
            (`<node>:<id>`, which is also its build job's handle), the machine and the build's phase. Then \
            watch it with `service_status {deploy, wait_s}` — one call that holds until the deploy is \
            proven, rolled back or failed — read its build with `repo_job_log {job: <deploy>}` and its \
            trial with `service_log {host, service}`, and stop it with `service_cancel {deploy}`. Only \
            origin/main is built: merge and push first. `motherd` is the daemon — models, Zen, npcd; \
            trying it stops every app on that machine for the trial. `web` is the gateway; `mcp` is the \
            connector. One machine per call: a new motherd joins the cluster as a candidate and the others \
            take it once it is certified, which is the user's decision (`cluster_command`); `web` and \
            `mcp` are deployed machine by machine. Say on each step where it runs — the machine and the \
            handle — so whoever watches knows. Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "minLength": 1, "maxLength": 32, "description": "The machine to deploy to, by name, as `cluster_status` and `repo_job_list` name it: `BlackWorld`, `WhiteWorld`."},
                "service": {"type": "string", "enum": SERVICES, "description": "`motherd` (the daemon), `web` (the gateway) or `mcp` (the connector)."}
            },
            "required": ["host", "service"],
            "additionalProperties": false
        }
    })
}

/// The job that builds `service` from origin/main and stages it.
pub fn build_job(service: &str) -> RunJob {
    RunJob {
        repo: REPO.to_string(),
        branch: BRANCH.to_string(),
        overlay: Snapshot::default(),
        program: "cargo".to_string(),
        args: ["build", "--release", "-p", service]
            .map(str::to_string)
            .to_vec(),
        timeout_secs: BUILD_TIMEOUT_S,
        stage: Some(format!("target/release/{service}.exe")),
    }
}

/// What a caller does next with deploy `h` of `service` to `host`.
pub fn next_steps(h: &Handle, host: &str, service: &str) -> String {
    format!(
        "Next: `service_status {{deploy: \"{h}\", wait_s: 240}}` waits for it and says where it is; \
         `repo_job_log {{job: \"{h}\"}}` summarises the build; once the trial starts, \
         `service_log {{host: \"{host}\", service: \"{service}\", mode: \"tail\"}}` reads it; \
         `service_cancel {{deploy: \"{h}\"}}` stops it."
    )
}

/// Wait for deploy `h`'s build and, once it has staged its binary, ask keep on
/// `host` for the trial. What this member does for a deploy ends there: the
/// trial is keep's, and its end is on keep's board.
async fn drive(call: OpsCall, agents: Agents, h: Handle, host: String, service: &'static str) {
    let at = h.at(&agents);
    let view = loop {
        match agents.wait(at, &h.id, STEP).await {
            Ok(v) if v.status.is_finished() => break v,
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(deploy = %h, host, service, "deploy: the build could not be read: {e}");
                return;
            }
        }
    };
    let artifact = match (&view.status, &view.staged) {
        (JobStatus::Exited { code: Some(0) }, Some(Staged::At(path))) => path.clone(),
        _ => {
            tracing::info!(deploy = %h, host, service, status = ?view.status, "deploy: the build did not stage");
            return;
        }
    };
    tracing::info!(deploy = %h, host, service, artifact, "deploy: trial");
    if let Err(e) = keep(
        &call,
        Method::POST,
        KEEP_TRIAL,
        &host,
        Some(&trial_body(service, &artifact, &h)),
    )
    .await
    {
        tracing::warn!(deploy = %h, host, service, "deploy: keep refused the trial: {e}");
    }
}

pub async fn run(call: &OpsCall, args: &Value) -> Answer {
    let (host, service) = match (string(args, "host"), string(args, "service").and_then(service_of)) {
        (Ok(h), Ok(s)) => (h.to_string(), s),
        (Err(e), _) | (_, Err(e)) => return Answer::error(e),
    };
    if call.worker {
        if let Err(why) = guard::check(call, &host, service).await {
            return Answer::error(why);
        }
    }
    let Some(agents) = call.session.agents() else {
        return Answer::error(NO_AGENTS);
    };
    let listed = match agents.listing().await {
        Ok(l) => l,
        Err(e) => return Answer::error(format!("the agents could not be listed: {e}")),
    };
    let node = match on_host(&listed, &host) {
        Ok(n) => n,
        Err(e) => return Answer::error(format!("{e}: a deploy builds on the machine it deploys to")),
    };
    let host = listed
        .iter()
        .find(|a| u64::from_str_radix(&a.node, 16) == Ok(node))
        .map_or(host, |a| a.host.clone());
    let (at, view) = match agents.submit(&call.lineage, build_job(service), Some(node)).await {
        Ok(started) => started,
        Err(e) => return Answer::error(format!("the build could not start on {host}: {e}")),
    };
    let node = agents.node_of(at).unwrap_or(node);
    let h = Handle::new(node, &view.id);
    tracing::info!(deploy = %h, host, service, "deploy: building");
    tokio::spawn(drive(call.clone(), agents, h.clone(), host.clone(), service));
    Answer::ok(format!(
        "Deploy {h}: {service} to {host}, building origin/main with `cargo build --release -p \
         {service}` — {}. When it has built, keep on {host} tries it: proven once it serves, or \
         rolled back.\n{}",
        phase(&view.phase, node),
        next_steps(&h, &host, service)
    ))
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    run(&OpsCall::of(ctx), args).await
}

#[cfg(test)]
mod tests {
    use super::super::built_service;
    use super::*;

    /// **The build is of main, with nothing laid over it**: anything else
    /// keep would refuse as a local build.
    #[test]
    fn the_job_builds_main_and_stages_the_service() {
        let job = build_job("web");
        assert_eq!((job.repo.as_str(), job.branch.as_str()), ("candle", "main"));
        assert_eq!(job.program, "cargo");
        assert_eq!(job.args, ["build", "--release", "-p", "web"]);
        assert_eq!(job.stage.as_deref(), Some("target/release/web.exe"));
        assert_eq!(job.overlay, Snapshot::default());
        let mut command = vec![job.program.clone()];
        command.extend(job.args);
        assert_eq!(built_service(&command), Some("web"));
    }

    #[test]
    fn the_next_steps_name_the_deploy_and_its_machine() {
        let next = next_steps(&Handle::new(0xab, "Xe"), "BlackWorld", "motherd");
        assert!(next.contains(r#"`service_status {deploy: "00000000000000ab:Xe", wait_s: 240}`"#));
        assert!(next.contains(r#"`service_log {host: "BlackWorld", service: "motherd", mode: "tail"}`"#));
    }

    /// The schema names the services keep runs, so the grammar offers only
    /// them, and the machine before the service.
    #[test]
    fn the_schema_offers_only_the_services_keep_runs() {
        let d = definition();
        assert_eq!(
            d["inputSchema"]["properties"]["service"]["enum"],
            json!(["motherd", "web", "mcp"])
        );
        assert_eq!(d["inputSchema"]["required"], json!(["host", "service"]));
    }
}
