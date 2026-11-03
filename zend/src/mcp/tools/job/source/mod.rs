//! Each kind of job, behind one face (`docs/jobs.md` §3): where it stands, its
//! output read as `LogAsk` asks, stopping it, and listing them.
//!
//! | Module | Kind |
//! |---|---|
//! | [`task`] | a worker of Zen's |
//! | [`build`] | a build or test run on an agent |
//! | [`deploy`] | a deploy: its build, then keep's trial |
//! | [`cluster`] | a maintenance command's job on a member |

use ::cluster::log_read::{LogAsk, LogRead};

use super::{JobHandle, Standing};
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::tools::Answer;

pub mod build;
pub mod cluster;
pub mod deploy;
pub mod task;

/// One job as `job_list` shows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub job: JobHandle,
    pub line: String,
    /// When it began, for ordering newest first.
    pub started_ms: u64,
}

/// Where `job` stands.
pub async fn standing(call: &OpsCall, job: &JobHandle) -> Result<Standing, String> {
    match job {
        JobHandle::Task { thread, name } => task::standing(call, job, thread, name),
        JobHandle::Build(h) => build::standing(call, job, h).await,
        JobHandle::Deploy(h) => deploy::standing(call, job, h).await,
        JobHandle::Cluster { node, n } => cluster::standing(call, job, node, *n).await,
    }
}

/// `ask` of `job`'s output.
pub async fn read(call: &OpsCall, job: &JobHandle, ask: &LogAsk) -> Result<LogRead, String> {
    match job {
        JobHandle::Task { thread, name } => task::read(call, thread, name, ask),
        JobHandle::Build(h) => build::read(call, h, ask).await,
        JobHandle::Deploy(h) => deploy::read(call, h, ask).await,
        JobHandle::Cluster { node, n } => cluster::read(call, node, *n, ask).await,
    }
}

/// Stop `job`.
pub async fn cancel(call: &OpsCall, job: &JobHandle) -> Answer {
    match job {
        JobHandle::Task { thread, name } => task::cancel(call, job, thread, name),
        JobHandle::Build(h) => build::cancel(call, job, h).await,
        JobHandle::Deploy(h) => deploy::cancel(call, job, h).await,
        JobHandle::Cluster { .. } => cluster::cancel(job),
    }
}

/// Every job of `kind` — every kind when `None` — the caller can see, and why
/// any kind could not be read.
pub async fn list(
    call: &OpsCall,
    kind: Option<&str>,
    running: bool,
    limit: usize,
) -> (Vec<Row>, Vec<String>) {
    let wants = |k: &str| kind.is_none_or(|w| w == k);
    let mut rows = Vec::new();
    let mut unread = Vec::new();
    if wants("task") {
        match task::list(call, running) {
            Ok(r) => rows.extend(r),
            Err(e) => unread.push(format!("tasks: {e}")),
        }
    }
    if wants("build") || wants("deploy") {
        match build::list(call, running, limit).await {
            Ok(r) => rows.extend(r.into_iter().filter(|r| wants(r.job.kind()))),
            Err(e) => unread.push(format!("builds and deploys: {e}")),
        }
    }
    if wants("cluster") && call.admin() {
        match cluster::list(call, running).await {
            Ok(r) => rows.extend(r),
            Err(e) => unread.push(format!("maintenance jobs: {e}")),
        }
    }
    rows.sort_by(|a, b| b.started_ms.cmp(&a.started_ms));
    rows.truncate(limit);
    (rows, unread)
}
