//! Sandbox jobs run in the background, their output in a log file.
//!
//! A [`SandboxServer`] is one repository's [`Sandbox`] with a job table in
//! front of it. [`SandboxServer::start_job`] starts a job and returns at once,
//! with:
//!
//! - **its id** — a random 64-bit number as URL-safe base64 ([`JobId`]);
//! - **its log** — `<jobs folder>/<id>.log`, which the command's output, both
//!   streams, is written to as it is printed;
//! - **its output as a stream** ([`OutputStream`]) — the same bytes, read
//!   from the log from its first byte and ending when the job does, for a
//!   caller that would rather follow the output than read the file;
//! - **its handle** ([`JobHandle`]) — which gives the job's outcome, and
//!   cancels the job when it is dropped first: the command and everything it
//!   started are killed.
//!
//! [`SandboxServer::query_job`] says where a job stands and how many lines its
//! log holds. The server keeps the last [`KEPT_JOBS`](table::KEPT_JOBS) jobs;
//! an older one is let go with its log, and asking after it is then
//! [`JobNotFound`].
//!
//! Each job sets aside whatever the checkout held and puts it back when it
//! ends — however it ends, a cancelled job included ([`Sandbox::run`]).
//!
//! The jobs folder sits outside every repository — the workspace folder's
//! [`JOBS_DIR`] — so no log is ever a file in a repository a job checks out,
//! captures or resets. One folder serves every repository's server; ids are
//! random, so their logs never meet.
//!
//! | Module | Concern |
//! |---|---|
//! | [`job_id`] | A job's id |
//! | [`status`] | Where a job stands, and what a query returns |
//! | [`log`] | The log file as it is written, and read back as a stream |
//! | [`table`] | The jobs kept, oldest let go first |

pub mod job_id;
pub mod log;
pub mod status;
pub mod table;

use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use tokio::task::JoinHandle;

use self::log::LogWriter;
use self::table::{Jobs, KEPT_JOBS};
use super::{Job, RunOutcome, Sandbox, SandboxCommand, SandboxError};
use crate::{BranchName, DiskWriteGrant, VfsStore};

pub use self::job_id::JobId;
pub use self::log::OutputStream;
pub use self::status::{JobInfo, JobNotFound, JobStatus};

/// The folder in the workspace folder that holds every job's log.
pub const JOBS_DIR: &str = "jobs";

/// A job to start.
pub struct JobRequest {
    /// The branch the conversation works on.
    pub branch: BranchName,
    /// The conversation's store over the repository, on `branch`.
    pub files: Arc<VfsStore>,
    pub command: SandboxCommand,
}

/// A job just started.
pub struct StartedJob {
    pub id: JobId,
    /// Its log file.
    pub log: PathBuf,
    /// Its output, from the first byte, as it is written.
    pub output: OutputStream,
    /// Its outcome — and dropping it first cancels the job.
    pub handle: JobHandle,
}

/// A running job's handle. Dropping it before the job ends cancels the job.
pub struct JobHandle {
    id: JobId,
    jobs: Arc<Mutex<Jobs>>,
    task: Option<JoinHandle<Result<RunOutcome, SandboxError>>>,
}

impl JobHandle {
    pub fn id(&self) -> &JobId {
        &self.id
    }

    /// Wait for the job to end, and take how it went. Abandoning the wait
    /// drops the handle, which cancels the job.
    pub async fn finish(mut self) -> Result<RunOutcome, SandboxError> {
        let task = self
            .task
            .as_mut()
            .expect("a handle holds its job until it ends");
        let ended = task.await;
        self.task = None;
        ended.map_err(|e| SandboxError::Interrupted(e.to_string()))?
    }
}

impl Drop for JobHandle {
    fn drop(&mut self) {
        let Some(task) = self.task.take() else {
            return;
        };
        if task.is_finished() {
            return;
        }
        task.abort();
        lock(&self.jobs).ended(&self.id, JobStatus::Cancelled);
    }
}

/// One repository's sandbox, running jobs in the background.
pub struct SandboxServer {
    sandbox: Arc<Sandbox>,
    jobs_dir: PathBuf,
    jobs: Arc<Mutex<Jobs>>,
}

impl SandboxServer {
    /// A server over `sandbox`, writing logs to `jobs_dir` — made if it does
    /// not exist.
    pub fn new(sandbox: Sandbox, jobs_dir: impl Into<PathBuf>) -> io::Result<Self> {
        Self::keeping(sandbox, jobs_dir, KEPT_JOBS)
    }

    fn keeping(sandbox: Sandbox, jobs_dir: impl Into<PathBuf>, kept: usize) -> io::Result<Self> {
        let jobs_dir = jobs_dir.into();
        std::fs::create_dir_all(&jobs_dir)?;
        Ok(Self {
            sandbox: Arc::new(sandbox),
            jobs_dir,
            jobs: Arc::new(Mutex::new(Jobs::new(kept))),
        })
    }

    pub fn sandbox(&self) -> &Sandbox {
        &self.sandbox
    }

    /// The folder the logs are written to.
    pub fn jobs_dir(&self) -> &Path {
        &self.jobs_dir
    }

    /// Start `request` in the background — see the module. Fails only when
    /// the log cannot be made; everything after that is the job's own
    /// outcome, in its status and its handle. Must be called within a Tokio
    /// runtime.
    pub fn start_job(&self, grant: DiskWriteGrant, request: JobRequest) -> io::Result<StartedJob> {
        std::fs::create_dir_all(&self.jobs_dir)?;
        let id = JobId::random();
        let log = self.jobs_dir.join(format!("{id}.log"));
        let (mut writer, written) = LogWriter::create(&log)?;
        let output = log::tail(&log, written.clone())?;
        let gone = lock(&self.jobs).insert(id.clone(), log.clone(), written);
        for old in gone {
            let _ = std::fs::remove_file(old);
        }

        let sandbox = Arc::clone(&self.sandbox);
        let jobs = Arc::clone(&self.jobs);
        let job_id = id.clone();
        let task = tokio::spawn(async move {
            let started = || lock(&jobs).running(&job_id);
            let job = Job {
                id: job_id.as_str(),
                branch: &request.branch,
                files: &request.files,
                command: &request.command,
                on_lock: Some(&started),
            };
            let result = sandbox.run(&grant, job, &mut writer).await;
            drop(writer);
            lock(&jobs).ended(&job_id, JobStatus::of(&result));
            result
        });
        Ok(StartedJob {
            id: id.clone(),
            log,
            output,
            handle: JobHandle {
                id,
                jobs: Arc::clone(&self.jobs),
                task: Some(task),
            },
        })
    }

    /// Where the job `id` stands, and how many lines its log holds.
    pub fn query_job(&self, id: &JobId) -> Result<JobInfo, JobNotFound> {
        lock(&self.jobs)
            .info(id)
            .ok_or_else(|| JobNotFound(id.clone()))
    }
}

/// The job table; a panic elsewhere never poisons it for good.
fn lock(jobs: &Mutex<Jobs>) -> MutexGuard<'_, Jobs> {
    jobs.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestRepo;
    use crate::{CommandPolicy, GitSource, Rev};

    /// **Past the jobs a server keeps, the oldest finished one is let go with
    /// its log**, and asking after it is then not found.
    #[tokio::test]
    async fn the_oldest_finished_job_goes_with_its_log() {
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        t.commit_all("base");
        let jobs_dir = tempfile::tempdir().unwrap();
        let (shell, args): (&str, [&str; 2]) = if cfg!(windows) {
            ("cmd", ["/C", "echo x"])
        } else {
            ("sh", ["-c", "echo x"])
        };
        let sandbox = Sandbox::new(t.repo(), CommandPolicy::allowing([shell]));
        let server = SandboxServer::keeping(sandbox, jobs_dir.path(), 1).unwrap();
        let main = BranchName::parse("main").unwrap();
        let files = Arc::new(VfsStore::on_branch(
            GitSource::open(&t.path).unwrap(),
            Rev::Branch(main.clone()),
        ));
        let request = || JobRequest {
            branch: main.clone(),
            files: Arc::clone(&files),
            command: SandboxCommand::new(shell).args(args),
        };

        let first = server
            .start_job(DiskWriteGrant::issue(), request())
            .unwrap();
        first.handle.finish().await.unwrap();
        assert!(first.log.is_file());
        let second = server
            .start_job(DiskWriteGrant::issue(), request())
            .unwrap();
        assert!(!first.log.exists(), "the first job's log was kept");
        assert_eq!(
            server.query_job(&first.id).unwrap_err(),
            JobNotFound(first.id.clone())
        );
        second.handle.finish().await.unwrap();
        assert!(server.query_job(&second.id).is_ok());
    }
}
