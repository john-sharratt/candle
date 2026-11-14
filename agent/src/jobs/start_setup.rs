//! [`Jobs::set_up`]: at start, each checkout runs the setup its repository
//! declares ([`crate::setup`]) — in order, one run job at a time, on the branch
//! the checkout has checked out, with no overlay — while a member's run on
//! that checkout waits behind it. A command that fails is said and the next
//! one runs; the setup's runs are kept and listed like any other.

use std::sync::Arc;
use std::time::Instant;

use tokio::task::spawn_blocking;
use zend_vfs::{JobStatus, Rev, VfsStore};

use cluster::reserve::machine as reservation;

use crate::overlay::Overlay;
use crate::setup::{parse_setup, SetupCommand, SETUP_FILE};
use crate::wire::RunJob;

use super::{Gate, JobError, Jobs, MAX_WAIT};

impl Jobs {
    /// Run every checkout's declared setup, each in a task of its own. A
    /// member's run on a checkout waits from this call until that checkout's
    /// setup has ended. Called once, as the agent starts, inside its runtime.
    pub fn set_up(self: &Arc<Self>) {
        let repos: Vec<String> = self.checkouts.all().map(|c| c.name.clone()).collect();
        for repo in repos {
            self.setting_up.begin(&repo);
            let jobs = Arc::clone(self);
            tokio::spawn(async move {
                jobs.set_up_checkout(&repo).await;
                jobs.setting_up.ended(&repo);
            });
        }
    }

    /// Until `repo`'s setup has ended; at once for a checkout with none.
    pub async fn setup_ended(&self, repo: &str) {
        self.setting_up.until_ended(repo).await;
    }

    async fn set_up_checkout(self: &Arc<Self>, repo: &str) {
        let (branch, commands) = match self.declared_setup(repo).await {
            Ok(Some(found)) => found,
            Ok(None) => return,
            Err(e) => {
                tracing::warn!(repo, "setup: {SETUP_FILE} could not be read: {e}");
                return;
            }
        };
        if commands.is_empty() {
            return;
        }
        if let Some(why) = reservation().why() {
            tracing::warn!(
                repo,
                why,
                "setup: skipped, the machine is reserved — it runs no builds"
            );
            return;
        }
        tracing::info!(
            repo,
            branch,
            commands = commands.len(),
            "setup: running the commands {SETUP_FILE} declares"
        );
        for command in commands {
            self.run_setup_command(repo, &branch, command).await;
        }
        tracing::info!(repo, "setup: done");
    }

    /// The branch `repo`'s checkout is on and the commands its setup file
    /// declares there, after a fetch of origin; `None` when the checkout is on
    /// no branch or the branch holds no setup file.
    async fn declared_setup(
        self: &Arc<Self>,
        repo: &str,
    ) -> Result<Option<(String, Vec<SetupCommand>)>, JobError> {
        if let Err(e) = self.fetch(repo).await {
            tracing::warn!(repo, "setup: reading it from what the checkout holds: {e}");
        }
        let jobs = Arc::clone(self);
        let name = repo.to_string();
        spawn_blocking(move || {
            let c = jobs.checkout(&name)?;
            let head = c
                .repo()
                .head()
                .map_err(|e| JobError::Bad(format!("reading the checkout's HEAD: {e}")))?;
            let Some(branch) = head.branch().cloned() else {
                return Ok(None);
            };
            let store = VfsStore::on_branch(Arc::clone(&c.source), Rev::Branch(branch.clone()));
            let text = store
                .read(SETUP_FILE)
                .map_err(|e| JobError::Bad(e.to_string()))?;
            let Some(text) = text else {
                return Ok(None);
            };
            let commands = parse_setup(&text).map_err(|e| JobError::Bad(e.to_string()))?;
            Ok(Some((branch.as_str().to_string(), commands)))
        })
        .await
        .map_err(|e| JobError::Bad(e.to_string()))?
    }

    /// Run one setup command to its end, saying how it went.
    async fn run_setup_command(self: &Arc<Self>, repo: &str, branch: &str, command: SetupCommand) {
        let line = command.line().join(" ");
        let run = RunJob {
            repo: repo.to_string(),
            branch: branch.to_string(),
            overlay: Overlay::default(),
            timeout_secs: command.timeout_secs(),
            program: command.program,
            args: command.args,
            stage: None,
        };
        let started = Instant::now();
        let view = match self.run(run, Gate::IsSetup).await {
            Ok(view) => view,
            Err(e) => {
                tracing::warn!(repo, command = %line, "setup: refused: {e}");
                return;
            }
        };
        let job = view.id;
        self.setting_up.running(repo, &job);
        tracing::info!(repo, job, command = %line, "setup: started");
        let status = loop {
            match self.wait(&job, MAX_WAIT).await {
                Ok(v) if v.status.is_finished() => break v.status,
                Ok(_) => {}
                Err(e) => break JobStatus::Failed { why: e.to_string() },
            }
        };
        let secs = started.elapsed().as_secs();
        if status == (JobStatus::Exited { code: Some(0) }) {
            tracing::info!(repo, job, command = %line, secs, "setup: ended");
        } else {
            tracing::warn!(
                repo,
                job,
                command = %line,
                secs,
                status = ?status,
                "setup: failed — the next command runs; read its log with job_read"
            );
        }
    }
}
