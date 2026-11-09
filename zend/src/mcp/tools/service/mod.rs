//! The `service_*` tools (`docs/jobs.md` §6): a service built from
//! origin/main on one machine and tried there through keep, the supervisor
//! that runs the machine's services, and what each keep runs. A deploy is a
//! job, watched, read and stopped with the `job_*` tools.
//!
//! | Module | Tool |
//! |---|---|
//! | [`deploy`] | `service_deploy` |
//! | [`status`] | `service_status` |
//! | [`guard`] | what a worker's deploy may not take down |
//!
//! **A deploy is named by its build job.** Its state is read from the agent
//! that builds it and from the keep that tries it, which records the handle on
//! the trial — never from the memory of the member that started it, so any
//! member answers for any deploy ([`resolve`]).

use std::fmt::Display;
use std::time::Duration;

use agent::client::Agents;
use agent::wire::{JobView, Staged};
use axum::http::Method;
use serde_json::{json, Value};
use zend_vfs::JobStatus;

use super::repo_job::{host_of, lost, state, Handle};
use crate::mcp::ops::{self, json_of};
use crate::mcp::ops_tools::OpsCall;

pub mod deploy;
pub mod guard;
pub mod status;

/// The services keep runs, each built from the package of its name.
pub const SERVICES: [&str; 4] = ["motherd", "web", "mcp", "yenbot"];

/// The workspace repository each service is built from: the cluster's from
/// candle, Yen Bots from its own (`docs/yenbot_app_host.md` §5.5).
pub fn repo_of(service: &str) -> &'static str {
    match service {
        "yenbot" => "yenbot",
        _ => "candle",
    }
}

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
                "`{text}` is not a service keep runs: `motherd` (the daemon), `web` (the gateway), \
                 `mcp` (the connector) or `yenbot` (Yen Bots)"
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

/// What a keep too old to record deploys is told to do.
pub const REBUILD_KEEP: &str = "rename the running `target\\release\\keep.exe`, `cargo build \
     --release -p keep`, then `curl -X POST http://127.0.0.1:8092/exit` on that machine — its \
     services keep running and the new keep adopts them";

/// keep `host`'s own build, from its `state`: `count-hash`. A keep that names
/// none predates recording each trial's deploy, so a deploy there could not find
/// the trial it asked for — it is refused, with what to do.
pub fn keep_build(host: &str, state: &Value) -> Result<String, String> {
    let Some(b) = state.get("keep").filter(|b| !b.is_null()) else {
        return Err(format!(
            "keep on {host} is too old to record which deploy a trial belongs to, so a deploy \
             there would build and try but never see its trial; update keep first: {REBUILD_KEEP}"
        ));
    };
    let local = b.get("local").and_then(Value::as_str).unwrap_or("");
    Ok(format!(
        "{}-{}{}",
        b.get("count").map_or_else(|| "?".into(), Value::to_string),
        b.get("hash").and_then(Value::as_str).unwrap_or("?"),
        if local.is_empty() {
            String::new()
        } else {
            format!(" (local {local})")
        }
    ))
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
    /// Its build job, or `None` once the agent that ran it no longer holds it.
    /// A motherd deploy always comes to that: the agent runs inside motherd,
    /// so the trial that starts the new motherd restarts the agent, and the
    /// job — its log with it — goes with the old process. The trial is still
    /// keep's, on keep's board, under the deploy's handle.
    pub build: Option<JobView>,
    pub stage: Stage,
}

/// The trial on `trials` — one service's board — that deploy `handle` asked
/// for: `Trying` while keep runs it, `Tried` once it ended.
fn trial_on(handle: &Handle, trials: Option<&Value>) -> Option<Stage> {
    let name = handle.to_string();
    let named = |key: &str| {
        trials
            .and_then(|t| t.get(key))
            .filter(|t| t.get("deploy").and_then(Value::as_str) == Some(name.as_str()))
            .cloned()
    };
    named("running")
        .map(Stage::Trying)
        .or_else(|| named("last").map(Stage::Tried))
}

/// The service and trial of deploy `handle` on keep's whole `state`, found by
/// the handle keep recorded on it: how a deploy whose build job is gone is
/// still answered for.
pub fn trial_in(handle: &Handle, state: &Value) -> Option<(&'static str, Stage)> {
    SERVICES
        .iter()
        .find_map(|s| trial_on(handle, trials_of(state, s)).map(|t| (*s, t)))
}

/// Where the deploy whose build is `view` stands, given keep's board for its
/// service on its machine (`trials`), at `now`.
pub fn stage_of(handle: &Handle, view: &JobView, trials: Option<&Value>, now: u64) -> Stage {
    if let Some(trial) = trial_on(handle, trials) {
        return trial;
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
        // Built, and its binary still being staged: the agent records where it
        // put it only once the copy is done, after the build has exited. Not
        // over, or a wait on it ends there and calls a good build failed.
        (JobStatus::Exited { code: Some(0) }, None) => Stage::Building,
        (status, _) => Stage::BuildFailed(state(status)),
    }
}

/// What a deploy found neither on its agent nor on keep's board answers.
pub fn untraced(handle: &Handle, host: &str, e: &impl Display) -> String {
    format!(
        "deploy:{handle}: its build job is not on the agent on {host} ({e}), and keep on {host} \
         names no trial of it, so nothing of it was tried: the agent restarted before the trial \
         was asked for. `service_status` says what {host} runs; deploy again"
    )
}

/// Deploy `handle` as it stands now: its build from the agent, its trial from
/// keep's board. A build job the agent no longer holds is answered from the
/// board alone ([`Deploy::build`]).
pub async fn resolve(
    call: &OpsCall,
    agents: &Agents,
    handle: &Handle,
    now: u64,
) -> Result<Deploy, String> {
    let build = match agents.view(handle.at(agents), &handle.id).await {
        Ok(b) => b,
        Err(e) => {
            let host = host_of(agents, handle.node).await;
            let state = keep(call, Method::GET, KEEP_STATE, &host, None)
                .await
                .map_err(|k| format!("{}; and {k}", lost(handle, &e)))?;
            let (service, stage) =
                trial_in(handle, &state).ok_or_else(|| untraced(handle, &host, &e))?;
            return Ok(Deploy {
                handle: handle.clone(),
                host,
                service,
                build: None,
                stage,
            });
        }
    };
    let service = built_service(&build.command).ok_or_else(|| {
        format!(
            "deploy:{handle} is not a deploy: it runs `{}`, not `cargo build --release -p \
             <service>` — it is build:{handle}",
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
        build: Some(build),
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

    /// **A keep names its own build, or is too old to deploy through**: one
    /// with no `keep` in its state records no deploy on its trials.
    #[test]
    fn a_keep_that_names_no_build_of_its_own_is_refused() {
        let new =
            json!({"keep": {"count": 4543, "hash": "1a2b3c4d5", "local": ""}, "services": []});
        assert_eq!(keep_build("BlackWorld", &new), Ok("4543-1a2b3c4d5".into()));
        let dirty = json!({"keep": {"count": 7, "hash": "ab", "local": "ffee"}});
        assert_eq!(
            keep_build("BlackWorld", &dirty),
            Ok("7-ab (local ffee)".into())
        );
        let old = json!({"services": []});
        assert_eq!(
            keep_build("WhiteWorld", &old),
            Err(format!(
                "keep on WhiteWorld is too old to record which deploy a trial belongs to, so a \
                 deploy there would build and try but never see its trial; update keep first: \
                 {REBUILD_KEEP}"
            ))
        );
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
        assert_eq!(
            built_service(&words("cargo build --release -p yenbot")),
            Some("yenbot")
        );
        assert_eq!(repo_of("yenbot"), "yenbot");
        assert_eq!(repo_of("web"), "candle");
        assert!(service_of("zend")
            .unwrap_err()
            .contains("not a service keep runs"));
    }

    /// **A deploy whose build job is gone is found on keep's board by its
    /// handle**, under whichever service it tried — a motherd deploy's agent
    /// restarts with the trial, taking the job with it. A trial of another
    /// deploy, or none, is not found.
    #[test]
    fn a_deploy_with_no_build_job_is_found_on_the_board() {
        let h = Handle::new(0x31f6, "0lt");
        let state = json!({"keep": {"count": 1, "hash": "a"}, "services": [
            {"name": "web", "trials": {"running": null,
                "last": {"proven": true, "deploy": "0000000000031f6:Other"}}},
            {"name": "motherd", "trials": {"last": null,
                "running": {"artifact": "C:/keep/stage/motherd.exe", "deploy": h.to_string(),
                    "started_ms": 7}}},
        ]});
        assert_eq!(
            trial_in(&h, &state),
            Some((
                "motherd",
                Stage::Trying(json!({"artifact": "C:/keep/stage/motherd.exe",
                    "deploy": "00000000000031f6:0lt", "started_ms": 7}))
            ))
        );
        assert_eq!(trial_in(&Handle::new(0x31f6, "Gone"), &state), None);
        assert_eq!(trial_in(&h, &json!({"services": []})), None);
        assert_eq!(
            untraced(&h, "WhiteWorld", &"404"),
            "deploy:00000000000031f6:0lt: its build job is not on the agent on WhiteWorld (404), \
             and keep on WhiteWorld names no trial of it, so nothing of it was tried: the agent \
             restarted before the trial was asked for. `service_status` says what WhiteWorld \
             runs; deploy again"
        );
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
        // Exited 0 with its binary not yet staged: still under way, not failed.
        let staging = build(ok.clone(), None, Some(5));
        let stage = stage_of(&h, &staging, None, 9);
        assert_eq!(stage, Stage::Building);
        assert!(!stage.is_over());
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
            trial_in(&h, &json!({"services": [{"name": "web", "trials": tried}]})),
            Some((
                "web",
                Stage::Tried(json!({"proven": true, "deploy": h.to_string()}))
            ))
        );
        assert_eq!(
            trial_body("web", "D:/s/web.exe", &h),
            json!({"service": "web", "artifact": "D:/s/web.exe", "deploy": "00000000000000ab:Xe"})
        );
    }
}
