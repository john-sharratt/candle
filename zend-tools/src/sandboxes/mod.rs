//! The workspace's command sandboxes — what `run_command` runs a program in.
//!
//! One [`SandboxServer`] per git repository of the workspace, each over the
//! repository's own folder and its own lock, so jobs on one repository run one
//! at a time and never wait on another's (`docs/zend_workspace_execution.md`
//! §7.4). A folder that is not a git repository — the uploads folder — has no
//! branch for a job to check out, and no sandbox. Every job's log is written
//! to the workspace folder's `jobs/`, outside every repository.
//!
//! A tool call waits for its job, which runs on a thread and runtime of its
//! own ([`block`]). Its output is read back a page at a time ([`page`]).
//!
//! | Module | Concern |
//! |---|---|
//! | [`block`] | Waiting on a job from a tool call's thread |
//! | [`page`] | A job's log, a page at a time |
//! | [`error`] | Why a job could not be run or read |

mod block;
mod error;
pub mod page;

pub use error::SandboxesError;
pub use page::{LogPage, PAGE_LINES};

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

use zend_vfs::{
    CommandPolicy, DiskWriteGrant, JobId, JobInfo, JobRequest, Repo, RunOutcome, Sandbox,
    SandboxServer, Workspace, JOBS_DIR,
};

/// A job that has ended.
#[derive(Debug)]
pub struct Ran {
    pub job: JobId,
    pub outcome: RunOutcome,
    /// Its log file — everything the command printed.
    pub log: PathBuf,
}

/// Every git repository's sandbox in one workspace.
pub struct Sandboxes {
    servers: BTreeMap<String, SandboxServer>,
    jobs_dir: PathBuf,
}

impl Sandboxes {
    /// A sandbox for each git repository of `workspace`, starting the
    /// programs `policy` lists, logging to the workspace folder's
    /// [`JOBS_DIR`] — made if it does not exist — each checkout first
    /// recovered from any job a crash cut short. A folder with no `.git` is
    /// not a repository and gets none; one with a `.git` that cannot be
    /// opened — unreadable, corrupt — is an error, not a quiet absence.
    pub fn for_workspace(workspace: &Workspace, policy: &CommandPolicy) -> io::Result<Self> {
        let jobs_dir = workspace.root().join(JOBS_DIR);
        let mut servers = BTreeMap::new();
        for repo in workspace.repos() {
            if !repo.dir.join(".git").exists() {
                continue;
            }
            let git = Repo::open(&repo.dir)
                .map_err(|e| io::Error::other(format!("repository {}: {e}", repo.name)))?;
            let sandbox = Sandbox::new(git, policy.clone());
            // A job a crash cut short left its checkout set aside — the
            // owner's branch, `HEAD` and files. It goes back now rather than
            // at the next job; one that cannot is left, named, for its owner
            // and refuses that next job the same way.
            if let Err(e) = sandbox.recover() {
                tracing::warn!("repository {}: {e}", repo.name);
            }
            servers.insert(repo.name.clone(), SandboxServer::new(sandbox, &jobs_dir)?);
        }
        Ok(Self { servers, jobs_dir })
    }

    /// The repositories a command can run in, by name.
    pub fn repos(&self) -> impl Iterator<Item = &str> {
        self.servers.keys().map(String::as_str)
    }

    /// The folder every job's log is written to.
    pub fn jobs_dir(&self) -> &Path {
        &self.jobs_dir
    }

    /// `Ok` when `repo` has a sandbox; otherwise why not, naming those that do.
    pub fn check(&self, repo: &str) -> Result<(), SandboxesError> {
        self.server(repo).map(|_| ())
    }

    fn server(&self, repo: &str) -> Result<&SandboxServer, SandboxesError> {
        self.servers
            .get(repo)
            .ok_or_else(|| SandboxesError::NoSandbox {
                repo: repo.to_string(),
                known: self.repos().collect::<Vec<_>>().join(", "),
            })
    }

    /// Run `request` in `repo`'s sandbox and wait for it to end.
    pub fn run(
        &self,
        grant: DiskWriteGrant,
        repo: &str,
        request: JobRequest,
    ) -> Result<Ran, SandboxesError> {
        let server = self.server(repo)?;
        block::on(async move {
            let started = server
                .start_job(grant, request)
                .map_err(SandboxesError::Log)?;
            let job = started.id.clone();
            let log = started.log.clone();
            // Nobody follows the output as it comes: the log holds it all.
            drop(started.output);
            let outcome = started.handle.finish().await?;
            Ok(Ran { job, outcome, log })
        })
    }

    /// Where the job `id` in `repo` stands.
    pub fn status(&self, repo: &str, id: &JobId) -> Result<JobInfo, SandboxesError> {
        Ok(self.server(repo)?.query_job(id)?)
    }

    /// Page `page` of the job `id`'s output, and where the job stands.
    pub fn output(
        &self,
        repo: &str,
        id: &JobId,
        page: usize,
    ) -> Result<(JobInfo, LogPage), SandboxesError> {
        let info = self.status(repo, id)?;
        let read = page::read(&info.log, page).map_err(SandboxesError::Log)?;
        Ok((info, read))
    }
}

#[cfg(test)]
mod tests;
