//! `service_deploy {host, service}`: build a service from origin/main on one
//! machine's agent and try it there through keep (`docs/jobs.md` §6). It
//! answers once the build job is accepted, with the deploy's handle,
//! `deploy:<node>:<id>`;
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
use crate::mcp::tools::job::JobHandle;
use crate::mcp::tools::repo_job::{on_host, phase, Handle, NO_AGENTS};
use crate::mcp::tools::{string, Answer, Ctx};

pub const NAME: &str = "service_deploy";

/// The repository every service is built from, and the one branch keep tries.
const REPO: &str = "candle";
const BRANCH: &str = "main";

/// How long a build may run: a first one on a machine compiles the workspace.
const BUILD_TIMEOUT_S: u64 = 3_600;

/// The daemon, and the connector released with it.
const DAEMON: &str = "motherd";
const CONNECTOR: &str = "mcp";

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
            (`deploy:<node>:<id>`), the machine and the build's phase. Then wait on it with `job_status` — \
            one call that holds until it is proven, rolled back or failed — whose pages are its build's \
            log then its trial's, read with `job_read` and searched with `job_grep`; stop it with \
            `job_cancel`, which rolls a trial back. Read `service_status` first to see what each machine \
            runs. Only \
            origin/main is built: merge and push first. `motherd` is the daemon — models, Zen, npcd; \
            trying it stops every app on that machine for the trial. `web` is the gateway; `mcp` is the \
            connector. One machine per call: a new motherd joins the cluster as a candidate and the others \
            take it once it is certified, which is the user's decision (`cluster_command`); `web` and \
            `mcp` are deployed machine by machine. Say on each step where it runs — the machine and the \
            handle — so whoever watches knows. Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "minLength": 1, "maxLength": 32, "description": "The machine to deploy to, by name, as `cluster_status` and `service_status` name it: `BlackWorld`, `WhiteWorld`."},
                "service": {"type": "string", "enum": SERVICES, "description": "`motherd` (the daemon), `web` (the gateway) or `mcp` (the connector)."}
            },
            "required": ["host", "service"],
            "additionalProperties": false
        }
    })
}

/// The job that builds `service` from origin/main and stages it. The daemon is
/// built with the connector, which is released with it: one build, so the
/// connector [`connector_job`] stages is the daemon's build.
pub fn build_job(service: &str) -> RunJob {
    let mut args: Vec<String> = ["build", "--release", "-p", service]
        .map(str::to_string)
        .to_vec();
    if service == DAEMON {
        args.extend(["-p", CONNECTOR].map(str::to_string));
    }
    RunJob {
        repo: REPO.to_string(),
        branch: BRANCH.to_string(),
        overlay: Snapshot::default(),
        program: "cargo".to_string(),
        args,
        timeout_secs: BUILD_TIMEOUT_S,
        stage: Some(format!("target/release/{service}.exe")),
    }
}

/// The job queued ahead of a daemon's deploy on the same agent: the same build
/// — the daemon and the connector — staging the connector, so the daemon the
/// trial starts finds its release's connector in keep's stage and follows it
/// (`cluster::upgrade::runtime::connector`), and offers it to the members that
/// install the release after it. Its packages are named `--package`, so it is
/// a build, never read as a deploy of its own.
pub fn connector_job() -> RunJob {
    RunJob {
        args: [
            "build",
            "--release",
            "--package",
            DAEMON,
            "--package",
            CONNECTOR,
        ]
        .map(str::to_string)
        .to_vec(),
        stage: Some(format!("target/release/{CONNECTOR}.exe")),
        ..build_job(CONNECTOR)
    }
}

/// What a caller does next with deploy `job`.
pub fn next_steps(job: &JobHandle) -> String {
    format!(
        "Next: `job_status {{jobs: [\"{job}\"], wait_s: 240}}` waits for it and says where it is; \
         `job_read {{job: \"{job}\", page: 1}}` reads its output — the build's log, then the \
         trial's; `job_cancel {{job: \"{job}\"}}` stops it."
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
    let (host, service) = match (
        string(args, "host"),
        string(args, "service").and_then(service_of),
    ) {
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
        Err(e) => {
            return Answer::error(format!("{e}: a deploy builds on the machine it deploys to"))
        }
    };
    let host = listed
        .iter()
        .find(|a| u64::from_str_radix(&a.node, 16) == Ok(node))
        .map_or(host, |a| a.host.clone());
    if service == DAEMON {
        if let Err(e) = agents
            .submit(&call.lineage, connector_job(), Some(node))
            .await
        {
            return Answer::error(format!(
                "the connector's build could not start on {host}, so the daemon is not deployed \
                 without it: {e}"
            ));
        }
    }
    let (at, view) = match agents
        .submit(&call.lineage, build_job(service), Some(node))
        .await
    {
        Ok(started) => started,
        Err(e) => return Answer::error(format!("the build could not start on {host}: {e}")),
    };
    let node = agents.node_of(at).unwrap_or(node);
    let h = Handle::new(node, &view.id);
    tracing::info!(deploy = %h, host, service, "deploy: building");
    let job = JobHandle::Deploy(h.clone());
    tokio::spawn(drive(call.clone(), agents, h, host.clone(), service));
    Answer::ok(format!(
        "Started {job}: {service} to {host}, building origin/main with `cargo build --release -p \
         {service}` — {}. When it has built, keep on {host} tries it: proven once it serves, or \
         rolled back.\n{}",
        phase(&view.phase, node),
        next_steps(&job)
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

    /// **The daemon is built with its connector**, and the job ahead of it
    /// stages the connector from that same build without reading as a deploy.
    #[test]
    fn the_daemon_is_built_with_the_connector_it_releases() {
        let daemon = build_job("motherd");
        assert_eq!(
            daemon.args,
            ["build", "--release", "-p", "motherd", "-p", "mcp"]
        );
        assert_eq!(daemon.stage.as_deref(), Some("target/release/motherd.exe"));
        let connector = connector_job();
        assert_eq!(
            connector.args,
            [
                "build",
                "--release",
                "--package",
                "motherd",
                "--package",
                "mcp"
            ]
        );
        assert_eq!(connector.stage.as_deref(), Some("target/release/mcp.exe"));
        assert_eq!(
            (connector.repo.as_str(), connector.branch.as_str()),
            ("candle", "main")
        );
        let words = |j: &RunJob| {
            let mut w = vec![j.program.clone()];
            w.extend(j.args.iter().cloned());
            w
        };
        assert_eq!(built_service(&words(&daemon)), Some("motherd"));
        assert_eq!(
            built_service(&words(&connector)),
            None,
            "a build, not a deploy"
        );
    }

    #[test]
    fn the_next_steps_name_the_deploy_and_its_machine() {
        let next = next_steps(&JobHandle::Deploy(Handle::new(0xab, "Xe")));
        assert!(
            next.contains(r#"`job_status {jobs: ["deploy:00000000000000ab:Xe"], wait_s: 240}`"#)
        );
        assert!(next.contains(r#"`job_cancel {job: "deploy:00000000000000ab:Xe"}`"#));
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
