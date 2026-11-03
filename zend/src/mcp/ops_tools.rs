//! The build and deploy tools, as one implementation two surfaces call
//! (`docs/build_deploy_tools.md` §6): Claude through the MCP connector, and a
//! worker of Zen's through its own tool round ([`OpsTools`], the daemon's
//! [`OpsRunner`]). Each tool runs against an [`OpsCall`] — who it is for and how
//! it may wait — so neither surface has a copy of its own to drift.

use std::sync::{Arc, OnceLock, Weak};

use axum::http::{HeaderName, HeaderValue};
use serde_json::Value;
use tokio::runtime::Handle;
use web::auth::Role;
use zend_tools::{run_watch, OpsRunner};

use super::roles::WorkerKind;
use super::threads;
use super::tools::{repo_job, service, Answer, Ctx};
use crate::session::ZendSession;

/// The longest a worker's call waits: it has no client to cut it off, and a
/// round that waited longer would hold the conversation for nothing.
const WORKER_WAIT_S: u64 = 300;

/// Who a build or deploy tool runs for, and how it may wait.
#[derive(Clone)]
pub struct OpsCall {
    pub session: Arc<ZendSession>,
    /// The lineage its jobs go under: an MCP caller's key, or a worker's
    /// conversation — what keeps its builds on the agent they warmed.
    pub lineage: String,
    /// Who keep is told the request is for: a person's identity, or nothing
    /// for a worker, which holds no person.
    pub headers: Vec<(HeaderName, HeaderValue)>,
    /// Where each step of a wait is said.
    pub progress: Option<Arc<dyn Fn(String) + Send + Sync>>,
    /// The most seconds a call may wait.
    pub ceiling: u64,
    /// A worker of Zen's: its deploys are guarded (§4).
    pub worker: bool,
}

impl OpsCall {
    /// The call an MCP tool call makes.
    pub fn of(ctx: &Ctx) -> OpsCall {
        let progress = ctx
            .progress
            .clone()
            .map(|tx| -> Arc<dyn Fn(String) + Send + Sync> {
                Arc::new(move |line| {
                    let _ = tx.send(line);
                })
            });
        OpsCall {
            session: Arc::clone(&ctx.session),
            lineage: ctx.caller.key(),
            headers: ctx.caller.headers(),
            progress,
            ceiling: ctx.mcp.client.wait_ceiling(),
            worker: false,
        }
    }

    /// Say `line` on the call's progress.
    pub fn say(&self, line: impl Into<String>) {
        if let Some(say) = &self.progress {
            say(line.into());
        }
    }

    /// The seconds a call waits: `wait_s` if given, `default` otherwise, at
    /// most the ceiling; and what it asked for when the ceiling cut it short.
    pub fn wait_of(&self, args: &Value, default: u64) -> (u64, Option<u64>) {
        let wanted = args
            .get("wait_s")
            .and_then(Value::as_u64)
            .unwrap_or(default);
        (
            wanted.min(self.ceiling),
            (wanted > self.ceiling).then_some(wanted),
        )
    }
}

/// The tools a worker of Zen's reaches through [`OpsRunner`], and whether
/// each changes something — a coder's — or only reads.
const WORKER_TOOLS: &[(&str, bool)] = &[
    (service::deploy::NAME, true),
    (service::cancel::NAME, true),
    (service::status::NAME, false),
    (service::log::NAME, false),
    (repo_job::status::NAME, false),
    (repo_job::log::NAME, false),
    (repo_job::list::NAME, false),
    (repo_job::cancel::NAME, true),
];

/// Why a worker of `role` doing `kind`'s work may not run `tool`.
fn unfit(tool: &str, role: Role, kind: WorkerKind) -> Option<String> {
    let Some((_, changes)) = WORKER_TOOLS.iter().find(|(name, _)| *name == tool) else {
        return Some(format!("`{tool}` is not a build or deploy tool"));
    };
    if !role.at_least(Role::Admin) {
        return Some(format!(
            "`{tool}` is an Admin's, and this task runs as `{role}`"
        ));
    }
    (*changes && kind != WorkerKind::Coder)
        .then(|| format!("`{tool}` is a coder's: this task changes nothing"))
}

/// Run build or deploy tool `name` for `call`.
pub async fn run(name: &str, call: &OpsCall, args: &Value) -> Answer {
    match name {
        repo_job::start::NAME => repo_job::start::run(call, args).await,
        repo_job::status::NAME => repo_job::status::run(call, args).await,
        repo_job::log::NAME => repo_job::log::run(call, args).await,
        repo_job::list::NAME => repo_job::list::run(call, args).await,
        repo_job::cancel::NAME => repo_job::cancel::run(call, args).await,
        service::deploy::NAME => service::deploy::run(call, args).await,
        service::status::NAME => service::status::run(call, args).await,
        service::log::NAME => service::log::run(call, args).await,
        service::cancel::NAME => service::cancel::run(call, args).await,
        other => Answer::error(format!("`{other}` is not a build or deploy tool")),
    }
}

/// The [`OpsRunner`] of one session: a worker's build and deploy tools.
#[derive(Default)]
pub struct OpsTools {
    /// The session it runs for, and the runtime its sockets live on: a call
    /// arrives on a tool round's own thread.
    bound: OnceLock<(Weak<ZendSession>, Handle)>,
}

impl OpsTools {
    /// Bind the runner to its session and that session's runtime.
    pub fn bind(&self, session: Weak<ZendSession>, runtime: Handle) {
        let _ = self.bound.set((session, runtime));
    }
}

impl OpsRunner for OpsTools {
    fn run(&self, caller: &str, tool: &str, args: Value) -> Result<String, String> {
        let Some((owner, thread, name)) = threads::conv_names(caller) else {
            return Err(format!(
                "`{tool}` is for a task Claude delegated through this MCP server, and this \
                 conversation is not one"
            ));
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
        if let Some(why) = unfit(tool, worker.role, worker.kind) {
            return Err(why);
        }
        let progress = run_watch::current().map(|watch| -> Arc<dyn Fn(String) + Send + Sync> {
            Arc::new(move |line| watch(line))
        });
        let call = OpsCall {
            session,
            lineage: caller.to_string(),
            headers: Vec::new(),
            progress,
            ceiling: WORKER_WAIT_S,
            worker: true,
        };
        let answer = runtime.block_on(run(tool, &call, &args));
        if answer.is_error {
            Err(answer.text)
        } else {
            Ok(answer.text)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A worker reads as an Admin and changes as an Admin's coder**; a tool
    /// outside the set is refused by name.
    #[test]
    fn a_worker_reads_as_an_admin_and_changes_as_a_coder() {
        assert_eq!(
            unfit(service::status::NAME, Role::Admin, WorkerKind::Reviewer),
            None
        );
        assert_eq!(
            unfit(service::deploy::NAME, Role::Admin, WorkerKind::Coder),
            None
        );
        assert!(
            unfit(service::deploy::NAME, Role::Admin, WorkerKind::Researcher)
                .unwrap()
                .contains("coder")
        );
        assert!(unfit(repo_job::log::NAME, Role::User, WorkerKind::Coder)
            .unwrap()
            .contains("Admin"));
        assert!(unfit("repo_job_start", Role::Admin, WorkerKind::Coder)
            .unwrap()
            .contains("not a build or deploy tool"));
    }
}
