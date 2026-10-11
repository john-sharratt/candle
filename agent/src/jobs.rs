//! The jobs an agent runs (§11.4), and what each came to.
//!
//! - **A run** fetches origin, so the base the overlay names is held here;
//!   builds the job's store — the checkout's branch at that base, with the
//!   overlay's changes over it — and starts the command in the checkout's
//!   sandbox: preserve → materialize → run → capture → restore
//!   (`zend_vfs::sandbox`). It answers at once; the outcome — exit code,
//!   output, the files the command changed as deltas — is kept for the
//!   member to read.
//! - **A push** fetches origin, takes the bundle's objects into the checkout,
//!   and pushes the specs from here, each under its lease, with the agent's
//!   own credentials. It is answered when it has been made.
//!
//! The last [`KEPT`] jobs are kept; asking after an older one is not found.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use tokio::sync::watch;
use tokio::task::{spawn_blocking, AbortHandle};
use tokio::time::timeout;
use zend_vfs::{
    BranchName, DiskWriteGrant, FetchSpec, JobId, JobRequest, JobStatus, Priority, RemoteName, Rev,
    RunOutcome, SandboxCommand, VfsStore, ORIGIN,
};

use crate::checkouts::{Checkout, Checkouts};
use crate::wire::{decode_bundle, JobView, PushJob, Pushed, RunJob, Submit};

/// Jobs kept after they end.
pub const KEPT: usize = 256;

/// The longest a member's wait on one job is held open.
pub const MAX_WAIT: Duration = Duration::from_secs(30);

/// Why a job was not started, or cannot be read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum JobError {
    #[error("this agent holds no repository `{0}`")]
    UnknownRepo(String),
    #[error("this agent cannot push to `{0}`'s origin")]
    NoPush(String),
    #[error("no job {0} on this agent")]
    NotFound(String),
    #[error("{0}")]
    Bad(String),
}

enum Kind {
    Run {
        /// The job's task: aborting it drops the job's handle, which cancels
        /// the job.
        task: AbortHandle,
        outcome: Option<RunOutcome>,
    },
    Push {
        status: JobStatus,
        pushed: Vec<Pushed>,
    },
}

struct Entry {
    repo: String,
    kind: Kind,
    ended: watch::Sender<bool>,
}

pub struct Jobs {
    checkouts: Checkouts,
    slots: u32,
    /// What its commands run at: below normal on a host that also runs a
    /// model member (§11.5).
    priority: Priority,
    running: AtomicU32,
    table: Mutex<Table>,
}

type Table = (HashMap<String, Entry>, VecDeque<String>);

fn keep_in(t: &mut Table, id: String, entry: Entry) {
    t.0.insert(id.clone(), entry);
    t.1.push_back(id);
    while t.1.len() > KEPT {
        if let Some(old) = t.1.pop_front() {
            t.0.remove(&old);
        }
    }
}

fn origin() -> RemoteName {
    RemoteName::parse(ORIGIN).expect("`origin` is a remote name")
}

impl Jobs {
    /// The jobs of `checkouts`, `slots` at a time, their commands run at
    /// `priority`.
    pub fn new(checkouts: Checkouts, slots: u32, priority: Priority) -> Arc<Jobs> {
        Arc::new(Jobs {
            checkouts,
            slots,
            priority,
            running: AtomicU32::new(0),
            table: Mutex::new((HashMap::new(), VecDeque::new())),
        })
    }

    pub fn checkouts(&self) -> &Checkouts {
        &self.checkouts
    }

    /// What its commands run at.
    pub fn priority(&self) -> Priority {
        self.priority
    }

    /// Job slots: all of them, and those running.
    pub fn slots(&self) -> (u32, u32) {
        (self.slots, self.running.load(Ordering::Relaxed))
    }

    fn table(&self) -> MutexGuard<'_, Table> {
        self.table.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn keep(&self, id: String, entry: Entry) {
        keep_in(&mut self.table(), id, entry);
    }

    fn checkout(&self, repo: &str) -> Result<&Checkout, JobError> {
        self.checkouts
            .get(repo)
            .ok_or_else(|| JobError::UnknownRepo(repo.to_string()))
    }

    /// Start `submit`: a run answers once started, a push once made.
    pub async fn submit(self: &Arc<Self>, submit: Submit) -> Result<JobView, JobError> {
        match submit {
            Submit::Run(run) => self.run(run).await,
            Submit::Push(push) => self.push(push).await,
        }
    }

    /// Fetch origin into `repo`'s checkout, so a base or a bundle's
    /// prerequisites origin holds are held here.
    async fn fetch(self: &Arc<Self>, repo: &str) -> Result<(), JobError> {
        if self.checkout(repo)?.origin.is_empty() {
            return Ok(());
        }
        let jobs = Arc::clone(self);
        let repo = repo.to_string();
        spawn_blocking(move || {
            let c = jobs.checkout(&repo)?;
            c.repo()
                .fetch(&origin(), &FetchSpec::AllBranches)
                .map(|_| ())
                .map_err(|e| JobError::Bad(format!("fetching `{repo}`'s origin: {e}")))
        })
        .await
        .map_err(|e| JobError::Bad(e.to_string()))?
    }

    async fn run(self: &Arc<Self>, run: RunJob) -> Result<JobView, JobError> {
        let c = self.checkout(&run.repo)?;
        let branch = BranchName::parse(&run.branch).map_err(|e| JobError::Bad(e.to_string()))?;
        self.fetch(&run.repo).await?;
        let store = VfsStore::on_branch(Arc::clone(&c.source), Rev::Branch(branch.clone()));
        store
            .restore(run.overlay)
            .map_err(|e| JobError::Bad(format!("the overlay: {e}")))?;
        let command = SandboxCommand::new(run.program)
            .args(run.args)
            .timeout(Duration::from_secs(run.timeout_secs))
            .priority(self.priority);
        let started = c
            .server
            .start_job(
                DiskWriteGrant::issue(),
                JobRequest {
                    branch,
                    files: Arc::new(store),
                    command,
                },
            )
            .map_err(|e| JobError::Bad(format!("the job's log: {e}")))?;
        let id = started.id.to_string();
        let (ended, _) = watch::channel(false);
        self.running.fetch_add(1, Ordering::Relaxed);
        let jobs = Arc::clone(self);
        let job = id.clone();
        let handle = started.handle;
        {
            // Held while the job is kept: a job ending at once records its
            // outcome only once it is in the table.
            let mut t = self.table();
            let task = tokio::spawn(async move {
                let outcome = handle.finish().await.ok();
                jobs.finished(&job, outcome);
            });
            keep_in(
                &mut t,
                id.clone(),
                Entry {
                    repo: run.repo,
                    kind: Kind::Run {
                        task: task.abort_handle(),
                        outcome: None,
                    },
                    ended,
                },
            );
        }
        self.view(&id)
    }

    /// Run `id` ended — by itself, or cancelled. Its slot is given back once,
    /// whichever comes first.
    fn finished(&self, id: &str, outcome: Option<RunOutcome>) {
        let mut t = self.table();
        let Some(e) = t.0.get_mut(id) else {
            self.running.fetch_sub(1, Ordering::Relaxed);
            return;
        };
        if *e.ended.borrow() {
            return;
        }
        if let Kind::Run { outcome: o, .. } = &mut e.kind {
            *o = outcome;
        }
        let _ = e.ended.send(true);
        self.running.fetch_sub(1, Ordering::Relaxed);
    }

    async fn push(self: &Arc<Self>, push: PushJob) -> Result<JobView, JobError> {
        let c = self.checkout(&push.repo)?;
        if !c.push {
            return Err(JobError::NoPush(push.repo));
        }
        let specs = push
            .specs
            .iter()
            .map(|l| l.spec())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| JobError::Bad(e.0))?;
        let bundle = push
            .bundle
            .as_deref()
            .map(decode_bundle)
            .transpose()
            .map_err(|e| JobError::Bad(e.0))?;
        self.fetch(&push.repo).await?;
        let jobs = Arc::clone(self);
        let repo = push.repo.clone();
        let made = spawn_blocking(move || {
            let c = jobs.checkout(&repo)?;
            if let Some(bundle) = &bundle {
                c.repo()
                    .take_bundle(bundle)
                    .map_err(|e| JobError::Bad(format!("the bundle: {e}")))?;
            }
            c.repo()
                .push_here(&origin(), &specs)
                .map_err(|e| JobError::Bad(e.to_string()))
        })
        .await
        .map_err(|e| JobError::Bad(e.to_string()))?;
        let (status, pushed) = match made {
            Ok(results) => (
                JobStatus::Exited { code: Some(0) },
                results.iter().map(Pushed::of).collect(),
            ),
            Err(e) => (JobStatus::Failed { why: e.to_string() }, Vec::new()),
        };
        let id = JobId::random().to_string();
        let (ended, _) = watch::channel(true);
        self.keep(
            id.clone(),
            Entry {
                repo: push.repo,
                kind: Kind::Push { status, pushed },
                ended,
            },
        );
        self.view(&id)
    }

    /// Where job `id` stands.
    pub fn view(&self, id: &str) -> Result<JobView, JobError> {
        let t = self.table();
        let e =
            t.0.get(id)
                .ok_or_else(|| JobError::NotFound(id.to_string()))?;
        Ok(match &e.kind {
            Kind::Push { status, pushed } => JobView {
                id: id.to_string(),
                repo: e.repo.clone(),
                status: status.clone(),
                lines: 0,
                bytes: 0,
                outcome: None,
                pushed: pushed.clone(),
            },
            Kind::Run { outcome, .. } => {
                let info = JobId::parse(id)
                    .and_then(|j| self.checkout(&e.repo).ok()?.server.query_job(&j).ok())
                    .ok_or_else(|| JobError::NotFound(id.to_string()))?;
                JobView {
                    id: id.to_string(),
                    repo: e.repo.clone(),
                    status: info.status,
                    lines: info.lines,
                    bytes: info.bytes,
                    outcome: outcome.clone(),
                    pushed: Vec::new(),
                }
            }
        })
    }

    /// Job `id` once it has ended, or as it stands after `wait`.
    pub async fn wait(&self, id: &str, wait: Duration) -> Result<JobView, JobError> {
        let mut ended = {
            let t = self.table();
            t.0.get(id)
                .ok_or_else(|| JobError::NotFound(id.to_string()))?
                .ended
                .subscribe()
        };
        let _ = timeout(wait.min(MAX_WAIT), ended.wait_for(|e| *e)).await;
        self.view(id)
    }

    /// Job `id`'s log file.
    pub fn log(&self, id: &str) -> Result<PathBuf, JobError> {
        let repo = {
            let t = self.table();
            t.0.get(id)
                .ok_or_else(|| JobError::NotFound(id.to_string()))?
                .repo
                .clone()
        };
        JobId::parse(id)
            .and_then(|j| self.checkout(&repo).ok()?.server.query_job(&j).ok())
            .map(|info| info.log)
            .ok_or_else(|| JobError::NotFound(id.to_string()))
    }

    /// Cancel job `id`: its command and everything it started are killed.
    pub fn cancel(&self, id: &str) -> Result<JobView, JobError> {
        let aborted = {
            let t = self.table();
            let e =
                t.0.get(id)
                    .ok_or_else(|| JobError::NotFound(id.to_string()))?;
            match &e.kind {
                Kind::Run { task, .. } => {
                    task.abort();
                    true
                }
                Kind::Push { .. } => false,
            }
        };
        if aborted {
            self.finished(id, None);
        }
        self.view(id)
    }
}
