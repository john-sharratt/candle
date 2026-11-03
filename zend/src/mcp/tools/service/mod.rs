//! The `service_*` deploy tools (`docs/build_deploy_tools.md` §4): a service
//! built from origin/main on one machine and tried there through keep, the
//! supervisor that runs the machine's services — and everything a caller does
//! with a deploy after: watch it, read the trial's log, stop it.
//!
//! | Module | Tool |
//! |---|---|
//! | [`deploy`] | `service_deploy` |
//! | [`status`] | `service_status` |
//! | [`log`] | `service_log` |
//! | [`cancel`] | `service_cancel` |
//! | [`guard`] | what a worker's deploy may not take down |
//!
//! **A deploy is its build job's handle.** Its state is read from the agent
//! that builds it and from the keep that tries it, which records the handle on
//! the trial — never from the memory of the member that started it, so any
//! member answers for any deploy ([`resolve`]).

use std::time::Duration;

use agent::client::Agents;
use agent::wire::{JobView, Staged};
use axum::http::Method;
use serde_json::{json, Value};
use zend_vfs::JobStatus;

use super::repo_job::{host_of, lost, state, Handle};
use crate::mcp::ops::{self, json_of};
use crate::mcp::ops_tools::OpsCall;

pub mod cancel;
pub mod deploy;
pub mod guard;
pub mod log;
pub mod status;

/// The services keep runs, each built from the package of its name.
pub const SERVICES: [&str; 3] = ["motherd", "web", "mcp"];

/// keep's routes.
pub const KEEP_STATE: &str = "/v1/keep/state";
pub const KEEP_TRIAL: &str = "/v1/keep/trial";
pub const KEEP_CANCEL: &str = "/v1/keep/trial/cancel";
pub const KEEP_LOG: &str = "/v1/keep/trial/log";

/// How long keep's answers are read for.
pub const KEEP_READ: Duration = Duration::from_secs(20);

/// How long a build may have ended with no trial of it before the member
/// that was to start one is taken to be gone.
pub const TRIAL_GRACE_MS: u64 = 60_000;

/// keep's ten minutes for a trial to print its proof line.
pub const PROOF_WINDOW_MS: u64 = 10 * 60_000;

/// `text` as a service keep runs.
pub fn service_of(text: &str) -> Result<&'static str, String> {
    SERVICES
        .iter()
        .copied()
        .find(|s| s.eq_ignore_ascii_case(text.trim()))
        .ok_or_else(|| {
            format!(
                "`{text}` is not a service keep runs: `motherd` (the daemon), `web` (the gateway) or \
                 `mcp` (the connector)"
            )
        })
}

/// The service a deploy's build job builds: the package after `-p` in
/// `cargo build --release -p <service>`.
pub fn built_service(command: &[String]) -> Option<&'static str> {
    let at = command.iter().position(|w| w == "-p")?;
    let built = command.get(at + 1)?;
    let is_build = command.first().is_some_and(|p| p == "cargo")
        && command.iter().any(|w| w == "build")
        && command.iter().any(|w| w == "--release");
    is_build.then(|| service_of(built).ok()).flatten()
}

/// Ask machine `host`'s keep `method path` with `body` for `call`.
pub async fn keep(
    call: &OpsCall,
    method: Method,
    path: &str,
    host: &str,
    body: Option<&Value>,
) -> Result<Value, String> {
    let reply = ops::keep_with(
        &call.session,
        call.headers(),
        method,
        path,
        host,
        body,
        KEEP_READ,
    )
    .await
    .map_err(|e| format!("keep on {host}: {e}"))?;
    json_of(&reply).map_err(|e| format!("keep on {host}: {e}"))
}

/// The trials of `service` in keep's `state`.
pub fn trials_of<'a>(state: &'a Value, service: &str) -> Option<&'a Value> {
    state
        .get("services")?
        .as_array()?
        .iter()
        .find(|s| s.get("name").and_then(Value::as_str) == Some(service))?
        .get("trials")
}

/// Where a deploy is.
#[derive(Debug, Clone, PartialEq)]
pub enum Stage {
    /// Its build runs.
    Building,
    /// Its build did not exit 0 and stage what it built.
    BuildFailed(String),
    /// Built and staged at the path, with no trial of it on keep's board
    /// yet; `stale` once that has lasted past [`TRIAL_GRACE_MS`].
    Built { artifact: String, stale: bool },
    /// keep is trying it: the board's running trial, as keep wrote it.
    Trying(Value),
    /// keep ended its trial: the board's last trial, as keep wrote it.
    Tried(Value),
}

impl Stage {
    /// Whether the deploy has ended, one way or another.
    pub fn is_over(&self) -> bool {
        matches!(
            self,
            Stage::BuildFailed(_) | Stage::Tried(_) | Stage::Built { stale: true, .. }
        )
    }
}

/// A deploy as it stands.
#[derive(Debug, Clone)]
pub struct Deploy {
    pub handle: Handle,
    pub host: String,
    pub service: &'static str,
    pub build: JobView,
    pub stage: Stage,
}

/// Where the deploy whose build is `view` stands, given keep's board for its
/// service on its machine (`trials`), at `now`.
pub fn stage_of(handle: &Handle, view: &JobView, trials: Option<&Value>, now: u64) -> Stage {
    let name = handle.to_string();
    let named = |t: &Value| t.get("deploy").and_then(Value::as_str) == Some(name.as_str());
    let on_board = |key: &str| {
        trials
            .and_then(|t| t.get(key))
            .filter(|t| !t.is_null() && named(t))
            .cloned()
    };
    if let Some(trial) = on_board("running") {
        return Stage::Trying(trial);
    }
    if let Some(trial) = on_board("last") {
        return Stage::Tried(trial);
    }
    if !view.status.is_finished() {
        return Stage::Building;
    }
    match (&view.status, &view.staged) {
        (JobStatus::Exited { code: Some(0) }, Some(Staged::At(artifact))) => Stage::Built {
            artifact: artifact.clone(),
            stale: view
                .ended_ms
                .is_some_and(|ended| now.saturating_sub(ended) > TRIAL_GRACE_MS),
        },
        (JobStatus::Exited { code: Some(0) }, Some(Staged::Not(why))) => {
            Stage::BuildFailed(format!("it built, but nothing was staged: {why}"))
        }
        (status, _) => Stage::BuildFailed(state(status)),
    }
}

/// Deploy `handle` as it stands now: its build from the agent, its trial from
/// keep's board.
pub async fn resolve(
    call: &OpsCall,
    agents: &Agents,
    handle: &Handle,
    now: u64,
) -> Result<Deploy, String> {
    let build = agents
        .view(handle.at(agents), &handle.id)
        .await
        .map_err(|e| lost(handle, &e))?;
    let service = built_service(&build.command).ok_or_else(|| {
        format!(
            "job {handle} is not a deploy: it runs `{}`, not `cargo build --release -p <service>` — \
             read it with `repo_job_status`",
            build.command.join(" ")
        )
    })?;
    let host = host_of(agents, handle.node).await;
    // A build still running has no trial yet: keep is asked only once it may.
    let trials = if build.status.is_finished() {
        let state = keep(call, Method::GET, KEEP_STATE, &host, None).await?;
        trials_of(&state, service).cloned()
    } else {
        None
    };
    let stage = stage_of(handle, &build, trials.as_ref(), now);
    Ok(Deploy {
        handle: handle.clone(),
        host,
        service,
        build,
        stage,
    })
}

/// The body that asks keep for deploy `handle`'s trial of `artifact` as
/// `service`.
pub fn trial_body(service: &str, artifact: &str, handle: &Handle) -> Value {
    json!({"service": service, "artifact": artifact, "deploy": handle.to_string()})
}

#[cfg(test)]
mod tests {
    use agent::wire::Phase;

    use super::*;

    fn build(status: JobStatus, staged: Option<Staged>, ended_ms: Option<u64>) -> JobView {
        JobView {
            id: "Xe".into(),
            repo: "candle".into(),
            status,
            lines: 9,
            bytes: 90,
            outcome: None,
            pushed: Vec::new(),
            fetched: None,
            listed: None,
            staged,
            phase: Phase::Ended,
            command: ["cargo", "build", "--release", "-p", "web"]
                .map(str::to_string)
                .to_vec(),
            started_ms: 0,
            ended_ms,
        }
    }

    #[test]
    fn a_deploys_service_is_the_package_it_builds() {
        let words = |s: &str| s.split(' ').map(str::to_string).collect::<Vec<_>>();
        assert_eq!(
            built_service(&words("cargo build --release -p motherd")),
            Some("motherd")
        );
        assert_eq!(built_service(&words("cargo test -p web")), None);
        assert_eq!(built_service(&words("cargo build -p web")), None);
        assert_eq!(built_service(&words("cargo build --release -p zend")), None);
        assert_eq!(service_of(" WEB "), Ok("web"));
        assert!(service_of("zend")
            .unwrap_err()
            .contains("not a service keep runs"));
    }

    /// **A deploy's stage is read from its build and keep's board**: building,
    /// failed, built and waiting for its trial — gone stale past the grace —
    /// trying, tried. A trial of another deploy is not this one's.
    #[test]
    fn a_deploys_stage_is_read_from_its_build_and_the_board() {
        let h = Handle::new(0xab, "Xe");
        let ok = JobStatus::Exited { code: Some(0) };
        let staged = Some(Staged::At("D:/keep/stage/web.exe".into()));
        let running = build(JobStatus::Running, None, None);
        assert_eq!(stage_of(&h, &running, None, 0), Stage::Building);
        let failed = build(JobStatus::Exited { code: Some(101) }, None, Some(5));
        assert_eq!(
            stage_of(&h, &failed, None, 9),
            Stage::BuildFailed("exited 101".into())
        );
        let built = build(ok.clone(), staged.clone(), Some(1_000));
        let other =
            json!({"running": {"artifact": "x", "deploy": "00000000000000ab:Other"}, "last": null});
        assert_eq!(
            stage_of(&h, &built, Some(&other), 30_000),
            Stage::Built {
                artifact: "D:/keep/stage/web.exe".into(),
                stale: false
            }
        );
        assert!(stage_of(&h, &built, Some(&other), 1_000 + TRIAL_GRACE_MS + 1).is_over());
        let trying = json!({"running": {"artifact": "x", "deploy": h.to_string(), "started_ms": 1}, "last": null});
        assert!(matches!(
            stage_of(&h, &built, Some(&trying), 0),
            Stage::Trying(_)
        ));
        let tried = json!({"running": null, "last": {"proven": true, "deploy": h.to_string()}});
        let stage = stage_of(&h, &built, Some(&tried), 0);
        assert!(matches!(stage, Stage::Tried(_)) && stage.is_over());
        assert_eq!(
            trial_body("web", "D:/s/web.exe", &h),
            json!({"service": "web", "artifact": "D:/s/web.exe", "deploy": "00000000000000ab:Xe"})
        );
    }
}
