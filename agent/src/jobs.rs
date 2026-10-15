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
//! - **A fetch** fetches origin and answers with what it holds for the
//!   branch asked — or every branch — and a bundle of the objects the member
//!   lacks, for a member whose repository lives with an agent (§10.3).
//!
//! The last [`KEPT`] jobs are kept; asking after an older one is not found.
//! Each is kept with its command line and its times, which is what an admin
//! watching the agent reads ([`Jobs::list`], [`Jobs::tail`]).

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::sync::watch;
use tokio::task::{spawn_blocking, AbortHandle};
use tokio::time::timeout;
use zend_vfs::{
    BranchName, DiskWriteGrant, FetchBundle, FetchSpec, JobId, JobRequest, JobStatus, Oid,
    Priority, RemoteName, Rev, RunOutcome, SandboxCommand, VfsStore, ORIGIN,
};

use crate::checkouts::{Checkout, Checkouts};
use crate::switch::Switch;
use crate::wire::{
    decode_bundle, AgentJobs, FetchJob, Fetched, JobKind, JobLine, JobView, LogChunk, PushJob,
    Pushed, RunJob, Submit,
};

/// Jobs kept after they end.
pub const KEPT: usize = 256;

/// The longest a member's wait on one job is held open.
pub const MAX_WAIT: Duration = Duration::from_secs(30);

/// The most of a log one [`Jobs::tail`] reads.
pub const TAIL_MAX: u64 = 256 * 1024;

/// Why a job was not started, or cannot be read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum JobError {
    #[error("this agent holds no repository `{0}`")]
    UnknownRepo(String),
    #[error("this agent cannot push to `{0}`'s origin")]
    NoPush(String),
    #[error("no job {0} on this agent")]
    NotFound(String),
    #[error("this agent is switched off")]
    Disabled,
    #[error("{0}")]
    Bad(String),
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
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
    Fetch {
        status: JobStatus,
    },
}

/// What a job runs and when, as [`Jobs::list`] shows it.
struct Line {
    branch: Option<String>,
    command: Vec<String>,
    started_ms: u64,
    ended_ms: Option<u64>,
}

struct Entry {
    repo: String,
    kind: Kind,
    line: Line,
    ended: watch::Sender<bool>,
}

pub struct Jobs {
    checkouts: Checkouts,
    slots: u32,
    /// What its commands run at: below normal on a host that also runs a
    /// model member (§11.5).
    priority: Priority,
    /// Whether it takes jobs (§11.7).
    switch: Switch,
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

/// The git command a push makes: one atomic push of every ref, each named
/// `<object>:<ref>`, or `:<ref>` for a delete.
fn push_command(push: &PushJob) -> Vec<String> {
    let mut command: Vec<String> = ["git", "push", "--atomic", ORIGIN].map(String::from).into();
    command.extend(push.specs.iter().map(|l| match &l.to {
        Some(to) => format!("{}:{}", &to[..to.len().min(12)], l.target),
        None => format!(":{}", l.target),
    }));
    command
}

/// The git command a fetch makes: origin's one branch, or all of them.
fn fetch_command(fetch: &FetchJob) -> Vec<String> {
    let mut command: Vec<String> = ["git", "fetch", ORIGIN].map(String::from).into();
    command.extend(fetch.branch.clone());
    command
}

/// Whether a fetch's answer `found` gives a member holding `have` nothing:
/// no objects to take, and every branch's commit one it already holds.
fn brought_nothing(found: &FetchBundle, have: &[Oid]) -> bool {
    found.bundle.is_none() && found.tips.values().all(|o| have.contains(o))
}

/// Up to `max` bytes of the file at `path` from byte `from`, and the file's
/// length.
fn read_from(path: &Path, from: u64, max: u64) -> io::Result<(Vec<u8>, u64)> {
    let mut file = File::open(path)?;
    let size = file.metadata()?.len();
    if from >= size {
        return Ok((Vec::new(), size));
    }
    file.seek(SeekFrom::Start(from))?;
    let mut bytes = Vec::with_capacity((size - from).min(max) as usize);
    file.take(max).read_to_end(&mut bytes)?;
    Ok((bytes, size))
}

impl Jobs {
    /// The jobs of `checkouts`, `slots` at a time, their commands run at
    /// `priority`; switched on or off as its jobs folder keeps it.
    pub fn new(checkouts: Checkouts, slots: u32, priority: Priority) -> Arc<Jobs> {
        let switch = Switch::open(checkouts.jobs_dir());
        Arc::new(Jobs {
            checkouts,
            slots,
            priority,
            switch,
            running: AtomicU32::new(0),
            table: Mutex::new((HashMap::new(), VecDeque::new())),
        })
    }

    /// Whether it takes jobs.
    pub fn enabled(&self) -> bool {
        self.switch.on()
    }

    /// Switch it on or off, and keep it so across a restart. The jobs it is
    /// running carry on.
    pub fn set_enabled(&self, on: bool) -> Result<(), JobError> {
        self.switch
            .set(on)
            .map_err(|e| JobError::Bad(format!("keeping the agent's switch: {e}")))
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

    /// Start `submit`: a run answers once started, a push once made, a fetch
    /// with what it found.
    pub async fn submit(self: &Arc<Self>, submit: Submit) -> Result<JobView, JobError> {
        if !self.enabled() {
            return Err(JobError::Disabled);
        }
        match submit {
            Submit::Run(run) => self.run(run).await,
            Submit::Push(push) => self.push(push).await,
            Submit::Fetch(fetch) => self.fetch_for(fetch).await,
        }
    }

    /// Fetch origin, and answer with what it holds for the job's branch — or
    /// every branch — and a bundle of the objects the member lacks. The
    /// answer is kept as the job's status alone: its bundle goes back once,
    /// with the submit.
    ///
    /// **A fetch that brings the member nothing is not kept.** A member
    /// following origin asks every minute (§10.3), and almost every answer is
    /// "as you were": kept, they filled the job table — what an admin reads
    /// of the agent, and the [`KEPT`] jobs it holds — with polls, pushing the
    /// runs that did something out of it. One that brought commits, or that
    /// failed, is kept like any job.
    async fn fetch_for(self: &Arc<Self>, fetch: FetchJob) -> Result<JobView, JobError> {
        self.checkout(&fetch.repo)?;
        let started_ms = now_ms();
        let command = fetch_command(&fetch);
        let spec = match &fetch.branch {
            Some(b) => {
                FetchSpec::Branch(BranchName::parse(b).map_err(|e| JobError::Bad(e.to_string()))?)
            }
            None => FetchSpec::AllBranches,
        };
        let have = fetch
            .have
            .iter()
            .map(|o| Oid::parse(o).map_err(|e| JobError::Bad(format!("`{o}`: {e}"))))
            .collect::<Result<Vec<_>, _>>()?;
        self.fetch(&fetch.repo).await?;
        let jobs = Arc::clone(self);
        let repo = fetch.repo.clone();
        let held = have.clone();
        let max_commits = fetch.max_commits.map(|n| n as usize);
        let found = spawn_blocking(move || {
            let c = jobs.checkout(&repo)?;
            c.repo()
                .bundle_for_fetch_sliced(&origin(), &spec, &held, max_commits)
                .map_err(|e| JobError::Bad(format!("bundling `{repo}`: {e}")))
        })
        .await
        .map_err(|e| JobError::Bad(e.to_string()))?;
        let id = JobId::random().to_string();
        let (status, fetched) = match found {
            Ok(f) if brought_nothing(&f, &have) => {
                return Ok(JobView {
                    id,
                    repo: fetch.repo,
                    status: JobStatus::Exited { code: Some(0) },
                    lines: 0,
                    bytes: 0,
                    outcome: None,
                    pushed: Vec::new(),
                    fetched: Some(Fetched::of(&f)),
                })
            }
            Ok(f) => (JobStatus::Exited { code: Some(0) }, Some(Fetched::of(&f))),
            Err(e) => (JobStatus::Failed { why: e.to_string() }, None),
        };
        let (ended, _) = watch::channel(true);
        self.keep(
            id.clone(),
            Entry {
                repo: fetch.repo,
                kind: Kind::Fetch {
                    status: status.clone(),
                },
                line: Line {
                    branch: None,
                    command,
                    started_ms,
                    ended_ms: Some(now_ms()),
                },
                ended,
            },
        );
        let mut view = self.view(&id)?;
        view.fetched = fetched;
        Ok(view)
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
        let started_ms = now_ms();
        let line = Line {
            branch: Some(run.branch.clone()),
            command: [run.program.clone()]
                .into_iter()
                .chain(run.args.iter().cloned())
                .collect(),
            started_ms,
            ended_ms: None,
        };
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
                    line,
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
        e.line.ended_ms = Some(now_ms());
        let _ = e.ended.send(true);
        self.running.fetch_sub(1, Ordering::Relaxed);
    }

    async fn push(self: &Arc<Self>, push: PushJob) -> Result<JobView, JobError> {
        let c = self.checkout(&push.repo)?;
        if !c.push {
            return Err(JobError::NoPush(push.repo));
        }
        let started_ms = now_ms();
        let command = push_command(&push);
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
                line: Line {
                    branch: None,
                    command,
                    started_ms,
                    ended_ms: Some(now_ms()),
                },
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
                fetched: None,
            },
            Kind::Fetch { status } => JobView {
                id: id.to_string(),
                repo: e.repo.clone(),
                status: status.clone(),
                lines: 0,
                bytes: 0,
                outcome: None,
                pushed: Vec::new(),
                fetched: None,
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
                    fetched: None,
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
                Kind::Push { .. } | Kind::Fetch { .. } => false,
            }
        };
        if aborted {
            self.finished(id, None);
        }
        self.view(id)
    }

    /// Whether it takes jobs, its slots, and up to `limit` of the jobs it
    /// keeps, newest first.
    pub fn list(&self, limit: usize) -> AgentJobs {
        let ids: Vec<String> = self.table().1.iter().rev().take(limit).cloned().collect();
        let (slots, running) = self.slots();
        AgentJobs {
            enabled: self.enabled(),
            slots,
            running,
            jobs: ids.iter().filter_map(|id| self.line(id).ok()).collect(),
        }
    }

    /// Job `id` as [`Self::list`] shows it.
    fn line(&self, id: &str) -> Result<JobLine, JobError> {
        let view = self.view(id)?;
        let t = self.table();
        let e =
            t.0.get(id)
                .ok_or_else(|| JobError::NotFound(id.to_string()))?;
        let kind = match &e.kind {
            Kind::Run { .. } => JobKind::Run,
            Kind::Push { .. } => JobKind::Push,
            Kind::Fetch { .. } => JobKind::Fetch,
        };
        Ok(JobLine {
            id: view.id,
            repo: view.repo,
            kind,
            branch: e.line.branch.clone(),
            command: e.line.command.clone(),
            status: view.status,
            lines: view.lines,
            bytes: view.bytes,
            started_ms: e.line.started_ms,
            ended_ms: e.line.ended_ms,
            changed: view.outcome.map_or(0, |o| o.changed.len()),
        })
    }

    /// Job `id`'s log from byte `from`, up to [`TAIL_MAX`] of it. A push or
    /// a fetch writes no log: its tail is empty, and ended.
    pub async fn tail(&self, id: &str, from: u64) -> Result<LogChunk, JobError> {
        let is_run = {
            let t = self.table();
            let e =
                t.0.get(id)
                    .ok_or_else(|| JobError::NotFound(id.to_string()))?;
            matches!(e.kind, Kind::Run { .. })
        };
        if !is_run {
            return Ok(LogChunk::of(from, &[], 0, true));
        }
        // Whether it had ended is read before the log is: an ended job's log
        // is then all there, and a reader that has caught up with it is done.
        let ended = self.view(id)?.status.is_finished();
        let path = self.log(id)?;
        let (bytes, size) = spawn_blocking(move || read_from(&path, from, TAIL_MAX))
            .await
            .map_err(|e| JobError::Bad(e.to_string()))?
            .map_err(|e| JobError::Bad(format!("job {id}'s log: {e}")))?;
        Ok(LogChunk::of(from, &bytes, size, ended))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::PushLine;

    /// **A push and a fetch are listed as the git command they make**: every
    /// ref of one atomic push, a delete as `:<ref>`.
    #[test]
    fn a_push_and_a_fetch_name_their_git_command() {
        let push = PushJob {
            repo: "candle".into(),
            bundle: None,
            specs: vec![
                PushLine {
                    target: "refs/heads/main".into(),
                    to: Some("0123456789abcdef0123456789abcdef01234567".into()),
                    expect: None,
                },
                PushLine {
                    target: "refs/tags/v1".into(),
                    to: None,
                    expect: Some("1111111111111111111111111111111111111111".into()),
                },
            ],
        };
        assert_eq!(
            push_command(&push),
            [
                "git",
                "push",
                "--atomic",
                "origin",
                "0123456789ab:refs/heads/main",
                ":refs/tags/v1"
            ]
        );
        let one = FetchJob {
            repo: "mind".into(),
            branch: Some("main".into()),
            have: Vec::new(),
            max_commits: None,
        };
        assert_eq!(fetch_command(&one), ["git", "fetch", "origin", "main"]);
        let all = FetchJob {
            branch: None,
            ..one
        };
        assert_eq!(fetch_command(&all), ["git", "fetch", "origin"]);
    }

    /// **A fetch brings nothing only when it has no objects to give and every
    /// branch it names is at a commit the member holds** — a new commit, or
    /// objects to take, make it worth keeping.
    #[test]
    fn a_fetch_brings_nothing_only_when_the_member_has_it_all() {
        let a = Oid::parse("1111111111111111111111111111111111111111").unwrap();
        let b = Oid::parse("2222222222222222222222222222222222222222").unwrap();
        let main = BranchName::parse("main").unwrap();
        let at = |o: &Oid, bundle: Option<Vec<u8>>| FetchBundle {
            tips: [(main.clone(), o.clone())].into(),
            bundle,
            partial: false,
        };
        assert!(brought_nothing(&at(&a, None), std::slice::from_ref(&a)));
        assert!(!brought_nothing(&at(&b, None), std::slice::from_ref(&a)));
        assert!(!brought_nothing(
            &at(&a, Some(b"PACK".to_vec())),
            std::slice::from_ref(&a)
        ));
        assert!(!brought_nothing(&at(&a, None), &[]));
    }

    /// A log is read from any byte, at most `max` of it, with its length; a
    /// read at or past its end is empty.
    #[test]
    fn a_log_is_read_from_a_byte() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.log");
        std::fs::write(&path, b"one\ntwo\nthree\n").unwrap();
        assert_eq!(
            read_from(&path, 0, 1024).unwrap(),
            (b"one\ntwo\nthree\n".to_vec(), 14)
        );
        assert_eq!(read_from(&path, 4, 3).unwrap(), (b"two".to_vec(), 14));
        assert_eq!(read_from(&path, 14, 1024).unwrap(), (Vec::new(), 14));
        assert_eq!(read_from(&path, 99, 1024).unwrap(), (Vec::new(), 14));
    }
}
