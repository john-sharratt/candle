//! Where `run_command` runs a program, and where a repository's pushes are
//! made (`docs/replicated_substrate.md` §11.5).
//!
//! A daemon on its own runs both on its own checkouts — the workspace's
//! [`Sandboxes`](crate::sandboxes::Sandboxes), and git pushing from the
//! repository's own folder. A daemon given `--cluster` runs both on agents:
//! the command on the agent the gateway picks, its changes coming back into
//! the conversation's store, and each push from an agent's checkout with the
//! agent's credentials. Which one a daemon has is its deployment, fixed at
//! startup.

use std::sync::Arc;

use zend_vfs::{DiskWriteGrant, JobId, JobInfo, JobRequest, Push};

use crate::sandboxes::{LogPage, Ran, SandboxesError};

pub trait Executor: Send + Sync {
    /// `Ok` when a command can run in `repo`; otherwise why not.
    fn check(&self, repo: &str) -> Result<(), SandboxesError>;

    /// Run `request` in `repo` for the conversation `lineage`, record what it
    /// changed in `request.files`, and wait for it to end.
    fn run(
        &self,
        grant: DiskWriteGrant,
        repo: &str,
        lineage: &str,
        request: JobRequest,
    ) -> Result<Ran, SandboxesError>;

    /// Page `page` of job `id`'s output, and where the job stands.
    fn output(
        &self,
        repo: &str,
        id: &JobId,
        page: usize,
    ) -> Result<(JobInfo, LogPage), SandboxesError>;

    /// What makes `repo`'s pushes.
    fn pusher(&self, repo: &str) -> Arc<dyn Push>;
}
