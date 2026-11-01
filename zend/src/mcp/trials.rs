//! Deploying from a worker: `service_deploy` (`docs/claude_zend_gateway.md`
//! §8.4).
//!
//! The steps an Admin takes by hand with `repo_job_start` and `service_trial`,
//! as one call: `origin/main` built on the machine's agent and staged there,
//! then keep's trial, read to its end. keep refuses anything but a build of
//! `origin/main` and restores the previous binary when one does not serve, so a
//! bad build cannot keep a member down; what is left to this module is what
//! keep cannot see:
//!
//! - **Whose worker it is.** Only a coder running as an Admin deploys.
//! - **Where it runs.** A worker runs inside `motherd`, and a trial stops the
//!   service it replaces: a worker trying `motherd` on its own machine would end
//!   its own call.
//! - **What it takes down.** A trial of `motherd` stops every app on the
//!   machine, so an app with no ready member elsewhere is refused. That
//!   decision is a person's, made with `service_trial`. A status that cannot be
//!   read refuses too.
//!
//! The runner reaches its session through a weak handle, as [`Parts`] does.
//!
//! [`Parts`]: super::parts::Parts

use std::sync::{Arc, OnceLock, Weak};
use std::time::{Duration, Instant};

use agent::client::{AgentError, JobWatch};
use agent::wire::{RunJob, Staged};
use axum::http::Method;
use host::cluster::up::send_up;
use serde_json::{json, Value};
use tokio::runtime::Handle;
use web::auth::Role;
use zend_tools::{run_watch, TrialRequest, TrialResponse, TrialRunner};
use zend_vfs::{JobStatus, Snapshot};

use super::ops::{self, json_of};
use super::roles::WorkerKind;
use super::threads;
use super::tools::repo_job::on_host;
use crate::session::ZendSession;
use crate::worker_hop;

/// The repository every service is built from.
const REPO: &str = "candle";
/// The one branch keep will try a build of.
const BRANCH: &str = "main";
/// How long a build may run: a first one on a machine compiles the workspace.
const BUILD_TIMEOUT_S: u64 = 3_600;
/// keep's ten minutes for the build to serve, and its time to restore one that
/// did not.
const TRIAL_DEADLINE: Duration = Duration::from_secs(15 * 60);
const POLL: Duration = Duration::from_secs(5);
/// How long keep's answers are read for.
const KEEP_READ: Duration = Duration::from_secs(60);
const STATE_READ: Duration = Duration::from_secs(20);

/// The `service_deploy` runner of one session.
#[derive(Default)]
pub struct Trials {
    /// The session it deploys for, and the runtime its sockets live on: a call
    /// arrives on a tool round's own thread.
    bound: OnceLock<(Weak<ZendSession>, Handle)>,
}

impl Trials {
    /// Bind the runner to its session and that session's runtime.
    pub fn bind(&self, session: Weak<ZendSession>, runtime: Handle) {
        let _ = self.bound.set((session, runtime));
    }
}

/// Why a worker of `role` doing `kind`'s work may not deploy.
fn unfit(role: Role, kind: WorkerKind) -> Option<String> {
    if !role.at_least(Role::Admin) {
        return Some(format!(
            "`service_deploy` is an Admin's, and this task runs as `{role}`"
        ));
    }
    (kind != WorkerKind::Coder)
        .then(|| "`service_deploy` is a coder's: this task changes nothing".to_string())
}

/// The job that builds `service` from `origin/main` and stages it.
fn build_job(service: &str) -> RunJob {
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

/// The ready tenants of the member on `host` in `status` — `/v1/status` as the
/// gateway gathers it — that no member on another machine has ready.
fn only_here(status: &Value, host: &str) -> Vec<String> {
    let rows = status.get("members").and_then(Value::as_array);
    let ready = |m: &Value| -> Vec<String> {
        m.pointer("/body/tenants")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
            .filter(|(_, t)| t.get("state").and_then(Value::as_str) == Some("ready"))
            .map(|(name, _)| name.clone())
            .collect()
    };
    let on = |m: &&Value| {
        m.get("host")
            .and_then(Value::as_str)
            .is_some_and(|h| h.eq_ignore_ascii_case(host))
    };
    let elsewhere: Vec<String> = rows
        .into_iter()
        .flatten()
        .filter(|m| !on(m))
        .flat_map(ready)
        .collect();
    let mut only: Vec<String> = rows
        .into_iter()
        .flatten()
        .filter(on)
        .flat_map(ready)
        .filter(|app| !elsewhere.contains(app))
        .collect();
    only.sort();
    only.dedup();
    only
}

/// Why a trial of `service` on `host` may not run from member `own`, with
/// `/v1/status` as `status`.
fn in_the_way(status: &Value, own: u64, host: &str, service: &str) -> Option<String> {
    if service != "motherd" {
        return None;
    }
    let own = format!("{own:016x}");
    let here = status
        .get("members")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|m| m.get("node").and_then(Value::as_str) == Some(own.as_str()))
        .and_then(|m| m.get("host"))
        .and_then(Value::as_str);
    let Some(here) = here else {
        return Some(
            "this member is not in the cluster's status, so what a trial would stop is \
                      unknown"
                .to_string(),
        );
    };
    if here.eq_ignore_ascii_case(host.trim()) {
        return Some(format!(
            "this task runs inside motherd on {here}, and a trial stops it: try it on another \
             machine, from a task running elsewhere"
        ));
    }
    let only = only_here(status, host.trim());
    (!only.is_empty()).then(|| {
        format!(
            "a trial on {host} takes {} out of service, with no ready member elsewhere: that is \
             the user's decision, so ask",
            only.join(", ")
        )
    })
}

/// How `service`'s last trial ended in keep's `state`, as `(ended_ms, proven,
/// detail)`.
fn last_trial(state: &Value, service: &str) -> Option<(u64, bool, String)> {
    let last = state
        .get("services")?
        .as_array()?
        .iter()
        .find(|s| s.get("name").and_then(Value::as_str) == Some(service))?
        .pointer("/trials/last")
        .filter(|t| !t.is_null())?;
    Some((
        last.get("ended_ms")?.as_u64()?,
        last.get("proven").and_then(Value::as_bool) == Some(true),
        last.get("detail")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
    ))
}

/// What a finished build job says: the staged path, or why there is none.
fn staged(status: &JobStatus, staged: Option<Staged>, tail: &str) -> Result<String, String> {
    match (status, staged) {
        (JobStatus::Exited { code: Some(0) }, Some(Staged::At(path))) => Ok(path),
        (JobStatus::Exited { code: Some(0) }, Some(Staged::Not(why))) => {
            Err(format!("the build ran but nothing was staged: {why}"))
        }
        (JobStatus::Exited { code: Some(0) }, None) => {
            Err("the build ran but nothing was staged".to_string())
        }
        (status, _) => Err(format!("the build failed ({status:?}):\n{tail}")),
    }
}

/// The last `max` characters of a log.
fn tail(log: &[u8], max: usize) -> String {
    let text = String::from_utf8_lossy(log);
    let skip = text.chars().count().saturating_sub(max);
    text.chars().skip(skip).collect()
}

impl Trials {
    async fn deploy(
        session: &Arc<ZendSession>,
        lineage: &str,
        req: &TrialRequest,
        say: impl Fn(String),
    ) -> Result<TrialResponse, String> {
        let (host, service) = (req.host.trim(), req.service);
        let own = worker_hop::own_node().ok_or("this member is in no cluster")?;
        let member = ops::member(session)?;
        let status = send_up(
            &member,
            Method::GET,
            "/v1/status",
            &[],
            Vec::new(),
            Some(STATE_READ),
        )
        .await
        .map_err(|e| format!("the cluster's status could not be read: {e:#}"))?;
        let status = json_of(&status).map_err(|e| format!("/v1/status: {e}"))?;
        if let Some(why) = in_the_way(&status, own, host, service) {
            return Err(why);
        }
        let keep = |method, path, body: Option<Value>, read_for| {
            let session = Arc::clone(session);
            let path: &'static str = path;
            async move {
                let reply = ops::keep_with(
                    &session,
                    Vec::new(),
                    method,
                    path,
                    host,
                    body.as_ref(),
                    read_for,
                )
                .await?;
                json_of(&reply)
            }
        };
        let state = keep(Method::GET, "/v1/keep/state", None, STATE_READ)
            .await
            .map_err(|e| format!("keep on {host}: {e}"))?;
        let before = last_trial(&state, service).map(|(ended, ..)| ended);

        let agents = session
            .agents()
            .ok_or("this member reaches no agent: it has no cluster and no agent of its own")?;
        let failed = |e: AgentError| format!("the build could not run: {e}");
        let listed = agents.listing().await.map_err(failed)?;
        let node = on_host(&listed, host)?;
        say(format!("building {service} from origin/main on {host}"));
        let watch = run_watch::current().map(|watch| -> JobWatch {
            Arc::new(move |view, last| watch(format!("{} lines, last: {last}", view.lines)))
        });
        let ran = agents
            .run_on(lineage, build_job(service), Some(node), watch)
            .await
            .map_err(failed)?;
        let artifact = staged(
            &ran.view.status,
            ran.view.staged.clone(),
            &tail(&ran.log, 2_000),
        )?;

        say(format!("trying {artifact} as {service} on {host}"));
        tracing::info!(lineage, host, service, artifact, "deploy: trial");
        keep(
            Method::POST,
            "/v1/keep/trial",
            Some(json!({"service": service, "artifact": artifact})),
            KEEP_READ,
        )
        .await
        .map_err(|e| format!("keep on {host}: {e}"))?;

        // keep's clock is not this member's, so the end of this trial is its
        // record replacing the one read before it began. Its answers stop
        // while a trial of the gateway restarts what carries them.
        let until = Instant::now() + TRIAL_DEADLINE;
        while Instant::now() < until {
            tokio::time::sleep(POLL).await;
            let Ok(state) = keep(Method::GET, "/v1/keep/state", None, STATE_READ).await else {
                continue;
            };
            if let Some((ended, proven, detail)) = last_trial(&state, service) {
                if Some(ended) != before {
                    return Ok(TrialResponse {
                        artifact,
                        proven,
                        detail,
                    });
                }
            }
        }
        Err(format!(
            "keep on {host} had not ended the trial of {artifact} after {} min: an Admin reads \
             it with `service_status`",
            TRIAL_DEADLINE.as_secs() / 60
        ))
    }
}

impl TrialRunner for Trials {
    fn run(&self, caller: &str, req: TrialRequest) -> Result<TrialResponse, String> {
        let Some((owner, thread, name)) = threads::conv_names(caller) else {
            return Err(
                "`service_deploy` is for a task Claude delegated through this MCP server, and \
                 this conversation is not one"
                    .into(),
            );
        };
        let (session, runtime) = self
            .bound
            .get()
            .and_then(|(s, rt)| Some((s.upgrade()?, rt.clone())))
            .ok_or("the model is not running")?;
        let records = session
            .shared_substrate()
            .ok_or("the model is still loading")?;
        let worker = threads::worker(&records, owner, thread, name)
            .ok_or_else(|| format!("the calling task `T:{thread}/{name}` has no record"))?;
        if let Some(why) = unfit(worker.role, worker.kind) {
            return Err(why);
        }
        let watch = run_watch::current();
        let say = move |line: String| {
            if let Some(watch) = &watch {
                watch(line);
            }
        };
        runtime.block_on(Self::deploy(&session, caller, &req, say))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HERE: u64 = 0xa;

    fn member(host: &str, node: u64, tenants: Value) -> Value {
        json!({"host": host, "node": format!("{node:016x}"), "status": 200,
               "body": {"tenants": tenants}})
    }

    fn cluster() -> Value {
        json!({"members": [
            member("blackworld", HERE, json!({"zend": {"state": "ready"}})),
            member("whiteworld", 0xb, json!({
                "zend": {"state": "ready"}, "npcd": {"state": "ready"}
            })),
            member("alienworld", 0xc, json!({"zend": {"state": "loading"}})),
        ]})
    }

    /// **A task never stops the motherd it runs in**, in any spelling of the
    /// machine's name.
    #[test]
    fn a_trial_of_its_own_machine_is_refused() {
        let why = in_the_way(&cluster(), HERE, " BlackWorld ", "motherd").unwrap();
        assert!(
            why.starts_with("this task runs inside motherd on blackworld"),
            "{why}"
        );
    }

    /// **An app with no ready member elsewhere stops the trial**, and is
    /// named; one that is only loading elsewhere does not count as ready.
    #[test]
    fn a_trial_that_takes_down_an_only_member_is_refused() {
        let why = in_the_way(&cluster(), HERE, "whiteworld", "motherd").unwrap();
        assert_eq!(
            why,
            "a trial on whiteworld takes npcd out of service, with no ready member elsewhere: \
             that is the user's decision, so ask"
        );
        assert_eq!(only_here(&cluster(), "alienworld"), Vec::<String>::new());
        let status = json!({"members": [
            member("blackworld", HERE, json!({"zend": {"state": "ready"}})),
            member("whiteworld", 0xb, json!({"zend": {"state": "ready"}})),
        ]});
        assert_eq!(in_the_way(&status, HERE, "whiteworld", "motherd"), None);
    }

    /// A member missing from the status refuses rather than guessing; the
    /// gateway and the connector hold no worker and no app.
    #[test]
    fn only_motherd_is_in_the_way_and_an_unknown_refuses() {
        assert!(in_the_way(&cluster(), 0xfff, "alienworld", "motherd")
            .unwrap()
            .contains("unknown"));
        assert_eq!(in_the_way(&cluster(), HERE, "blackworld", "web"), None);
        assert_eq!(in_the_way(&cluster(), HERE, "blackworld", "mcp"), None);
    }

    #[test]
    fn only_an_admins_coder_deploys() {
        assert_eq!(unfit(Role::Admin, WorkerKind::Coder), None);
        assert!(unfit(Role::User, WorkerKind::Coder)
            .unwrap()
            .contains("Admin"));
        assert!(unfit(Role::Admin, WorkerKind::Reviewer)
            .unwrap()
            .contains("coder"));
        assert!(unfit(Role::Admin, WorkerKind::Researcher)
            .unwrap()
            .contains("coder"));
    }

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
    }

    #[test]
    fn a_trial_is_read_from_its_service_alone() {
        let state = json!({"services": [
            {"name": "web", "trials": {"last": {"ended_ms": 1, "proven": true, "detail": "w"}}},
            {"name": "motherd", "trials": {"last": {"ended_ms": 9, "proven": false, "detail": "m"}}},
            {"name": "mcp", "trials": {"last": null}},
        ]});
        assert_eq!(last_trial(&state, "motherd"), Some((9, false, "m".into())));
        assert_eq!(last_trial(&state, "mcp"), None);
    }

    #[test]
    fn a_build_answers_its_staged_path_or_why_not() {
        let ok = JobStatus::Exited { code: Some(0) };
        assert_eq!(
            staged(&ok, Some(Staged::At("D:/stage/a.exe".into())), ""),
            Ok("D:/stage/a.exe".into())
        );
        assert!(staged(&ok, Some(Staged::Not("no file".into())), "")
            .unwrap_err()
            .contains("no file"));
        let failed = staged(&JobStatus::Exited { code: Some(101) }, None, "error[E0308]");
        assert!(failed.unwrap_err().contains("error[E0308]"));
    }
}
