//! Running a command line program on a repository, as one conversation.
//!
//! A [`Sandbox`] is one repository's: its working folder is the repository's
//! own checkout, which belongs to the sandbox and to nothing else — every
//! conversation reads the repository through its branch in git
//! ([`crate::vfs`]), never through this folder. A conversation's changes to the
//! repository live as deltas in its [`VfsStore`]; a program run on the machine
//! — a build, a test run, a formatter — needs them as files, and what it
//! changes has to come back into the store. [`Sandbox::run`] is that whole
//! round trip for one [`Job`], one job at a time:
//!
//! 1. **Lock** the checkout. A second job on the same repository waits here.
//! 2. **Set aside** what the checkout holds — the folder may be someone's
//!    working copy, with a branch checked out and work not yet committed:
//!    `HEAD` recorded, the index and every changed file snapshotted byte for
//!    byte under refs of the layer's own, and a journal written that a
//!    crash can be recovered from ([`preserve`]);
//! 3. **put the checkout on the conversation's branch**, at the commit the
//!    conversation's files are based on — where origin already holds it, the
//!    local branch first brought up to it when it lags, or moved onto it for
//!    the job when it holds commits of its owner's origin never had, which
//!    step 9 puts back (`follow`);
//! 4. **apply** the conversation's changes onto it, writing only the files
//!    whose bytes are wrong (steps 3 and 4 are [`materialize`]);
//! 5. **check** the command ([`CommandPolicy`]). Every rule that reads the
//!    command line alone — git run directly, the allow-list, paths out of the
//!    repository — is checked before step 1, so a refused command touches
//!    nothing; the one that reads the checkout — a program the conversation
//!    wrote into the repository is a plain file there — is checked here,
//!    where it now stands;
//! 6. **run** it, its output written to the job's sink as it comes
//!    ([`process`]);
//! 7. **diff** — put the branch and `HEAD` back where step 3 left them, should
//!    the command have committed, switched or deleted the branch, and read
//!    back what it changed, commits included, as deltas against the
//!    conversation's own state ([`capture`]);
//! 8. **record** the deltas in the conversation's store;
//! 9. **put back** what step 2 set aside — `HEAD` where it was, the
//!    snapshot's files and index written back, then let go — then
//!    **unlock**, and hand back how the command
//!    ended and the files it changed ([`RunOutcome`]).
//!
//! Step 9 happens however the job ends: a job that fails, panics or is
//! abandoned part way puts the checkout back as it goes, before the lock is
//! let go. Ignored files — build outputs — are never set aside or removed
//! beyond the paths the job writes or its branch adds, so the next job,
//! whichever conversation it is, finds a warm build cache.
//!
//! [`preserve`]: crate::checkout::preserve()
//!
//! | Module | Concern |
//! |---|---|
//! | [`command`] | The program, its arguments and its timeout |
//! | `follow` | Putting the local branch on the conversation's base: a lagging one brought up, unpushed commits set aside |
//! | [`policy`] | The security check a command passes before it runs |
//! | `git_use` | Finding git run directly — as the program, or in a shell's script |
//! | [`process`] | Starting it, passing on its output, killing its process tree |
//! | `resolve` | Finding the file a program name on the `PATH` starts |
//! | [`outcome`] | What a run hands back |
//! | [`server`] | Jobs run in the background, their output in a log file |
//!
//! [`materialize`]: crate::checkout::materialize()
//! [`capture`]: crate::checkout::capture()

pub mod command;
mod error;
mod follow;
#[cfg(test)]
mod follow_tests;
mod git_use;
pub mod outcome;
pub mod policy;
pub mod process;
mod resolve;
pub mod server;

use std::path::Path;
use std::sync::Arc;

use tokio::io::AsyncWrite;
use tokio::sync::{Mutex, OwnedMutexGuard};

use crate::checkout::{self, CheckoutError, Ledger, Preserved};
use crate::file_delta::TimedDelta;
use crate::{BranchName, DiskWriteGrant, FileChanges, Oid, Repo, Rev, VfsStore};

pub use command::{SandboxCommand, DEFAULT_TIMEOUT};
pub use error::SandboxError;
pub use outcome::{ChangedFile, Output, RunOutcome, Unrecorded};
pub use policy::{CommandPolicy, Refused};
pub use server::{
    JobHandle, JobId, JobInfo, JobNotFound, JobRequest, JobStatus, OutputStream, SandboxServer,
    StartedJob, JOBS_DIR,
};

/// The checkout, held by one job: its own state set aside, what the job put
/// on it, and the lock that keeps every other job off it.
///
/// Fields drop in order, so a session dropped part way — a job that failed,
/// panicked or was abandoned — puts the checkout's own state back first, and
/// only then lets the next job in.
struct Session {
    preserved: Preserved,
    /// What the job put on the checkout beyond its branch's commit.
    ledger: Option<Ledger>,
    /// Held, never read: dropping it lets the next job in.
    _lock: OwnedMutexGuard<()>,
}

/// One job: a command, run as one conversation.
pub struct Job<'a> {
    /// This job's id, which the checkout's set-aside state is journalled
    /// under.
    pub id: &'a str,
    /// The branch the conversation works on.
    pub branch: &'a BranchName,
    /// The conversation's store over this repository.
    pub files: &'a VfsStore,
    pub command: &'a SandboxCommand,
    /// Told when the job has the checkout — when it stops waiting on another.
    pub on_lock: Option<&'a (dyn Fn() + Sync)>,
}

/// One repository's sandbox: its checkout, the programs it may run there, and
/// the lock that keeps jobs on it one at a time.
pub struct Sandbox {
    repo: Arc<Repo>,
    policy: CommandPolicy,
    checkout: Arc<Mutex<()>>,
}

impl Sandbox {
    pub fn new(repo: Repo, policy: CommandPolicy) -> Self {
        Self {
            repo: Arc::new(repo),
            policy,
            checkout: Arc::new(Mutex::new(())),
        }
    }

    pub fn repo(&self) -> &Repo {
        &self.repo
    }

    pub fn policy(&self) -> &CommandPolicy {
        &self.policy
    }

    /// Put back whatever a job that is gone — a crash, a restore that
    /// failed — left set aside in the checkout: its owner's branch, `HEAD`
    /// and files. Made when the sandbox is brought up, so a crash never leaves
    /// someone's checkout set aside until the next job; the next job makes it
    /// too. An error names what is still set aside, and where.
    pub fn recover(&self) -> Result<(), CheckoutError> {
        checkout::recover(&self.repo)
    }

    /// Run `job`, writing what its command prints to `output`, and record
    /// what it changed in the job's store. See the module for each step.
    ///
    /// The store must be the conversation's overlay over this repository, on
    /// the job's branch at the commit it holds. Whatever the checkout held
    /// before — anyone's own work in it — is set aside first and put back
    /// last, whether the job succeeds, fails or is abandoned part way (its
    /// future dropped, which kills the command); nothing else is let onto the
    /// checkout until it is back.
    pub async fn run(
        &self,
        _grant: &DiskWriteGrant,
        job: Job<'_>,
        output: &mut (dyn AsyncWrite + Unpin + Send),
    ) -> Result<RunOutcome, SandboxError> {
        let base = self.check_store(job.files, job.branch)?;
        // 5, for everything that needs no checkout: a refused command never
        // takes the lock.
        self.policy.check_command(job.command)?;
        // 1. Lock.
        let lock = Arc::clone(&self.checkout).lock_owned().await;
        if let Some(on_lock) = job.on_lock {
            on_lock();
        }
        let changes = job
            .files
            .changes()
            .map_err(|e| SandboxError::Store(e.to_string()))?;
        // 2. Set the checkout's own state aside.
        let repo = Arc::clone(&self.repo);
        let why = format!("sandbox job {} ran", job.id);
        let writes: Vec<String> = changes.paths().map(str::to_string).collect();
        let target = base.clone();
        let session = tokio::task::spawn_blocking(move || {
            checkout::preserve(repo, &why, &writes, Some(&target)).map(|preserved| Session {
                preserved,
                ledger: None,
                _lock: lock,
            })
        })
        .await
        .map_err(|e| SandboxError::Interrupted(e.to_string()))??;
        // 3, 4. The branch, and the conversation's changes over it.
        let session = self.put_on(session, job.branch, &base, changes).await?;
        // 5. Check what reads the checkout.
        self.policy
            .check_on_checkout(self.repo.dir(), job.command)?;
        // 6. Run.
        let executed = process::execute(self.repo.dir(), job.command, output)
            .await
            .map_err(|source| SandboxError::Start {
                program: job.command.program.clone(),
                source,
            })?;
        // 7. Diff.
        let repo = Arc::clone(&self.repo);
        let branch = job.branch.clone();
        let (session, captured) = off_runtime(session, move |session| {
            let ledger = session
                .ledger
                .as_mut()
                .expect("the checkout was just put in place");
            checkout::capture(&repo, &branch, ledger)
        })
        .await?;
        // 8. Record.
        let (changed, unrecorded) = record(job.files, captured);
        // 9. Put the checkout's own state back, then unlock.
        let (session, ()) = off_runtime(session, |session| session.preserved.restore()).await?;
        drop(session);
        Ok(RunOutcome {
            exit_code: executed.exit_code,
            timed_out: executed.timed_out,
            output: executed.output,
            changed,
            unrecorded,
        })
    }

    /// Put the checkout on `branch` with `changes` laid over it. The changes
    /// are made on `base`: a branch that moved off it before the checkout was
    /// put in place fails the job.
    ///
    /// The local branch is put on `base` here where origin already holds it
    /// ([`follow::follow`]) — under the checkout's lock and after
    /// [`preserve`] recorded where the owner's `HEAD` stood. A branch behind
    /// it is brought up to it, so the owner's checkout comes back as a
    /// branch that only moved on, its own changes carried onto it, never as
    /// uncommitted work reverting what origin gained; a branch holding
    /// commits origin never had is set aside and comes back at them.
    ///
    /// [`preserve`]: crate::checkout::preserve()
    async fn put_on(
        &self,
        session: Session,
        branch: &BranchName,
        base: &Oid,
        changes: FileChanges,
    ) -> Result<Session, SandboxError> {
        let repo = Arc::clone(&self.repo);
        let branch = branch.clone();
        let base = base.clone();
        let (session, ()) = off_runtime(session, move |session| {
            follow::follow(&repo, &mut session.preserved, &branch, &base)?;
            let grant = DiskWriteGrant::issue();
            let (ledger, _) = checkout::materialize(&repo, &grant, &branch, &changes)?;
            let at_base = ledger.base() == base.as_str();
            session.ledger = Some(ledger);
            if !at_base {
                return Err(CheckoutError::BranchMoved {
                    branch: branch.to_string(),
                });
            }
            Ok(())
        })
        .await?;
        Ok(session)
    }

    /// `files` is an overlay over this sandbox's repository, reading
    /// `branch` at the commit the branch holds — the base its changes are
    /// made on, which the checkout is put on. Returns that commit.
    fn check_store(&self, files: &VfsStore, branch: &BranchName) -> Result<Oid, SandboxError> {
        let same = files
            .root()
            .is_some_and(|root| same_folder(root, self.repo.dir()));
        if !same {
            return Err(SandboxError::WrongStore {
                store: files.root().map(Path::to_path_buf),
                repo: self.repo.dir().to_path_buf(),
            });
        }
        let reads = files.rev();
        if reads != Some(Rev::Branch(branch.clone())) {
            return Err(SandboxError::WrongBranch {
                branch: branch.to_string(),
                reads: reads.map(|r| format!("{r:?}")),
            });
        }
        let store = |e: String| SandboxError::Store(e);
        let base = files.base().map_err(|e| store(e.to_string()))?;
        // Commits on the local branch that origin never had are its owner's,
        // not on the record the conversation pinned: they are set aside for
        // the job and put back after it (`put_on`), and the job is judged
        // against the rest of the branch.
        let tip = self
            .repo
            .local_branch(branch)
            .map_err(|e| store(e.to_string()))?
            .and_then(|local| local.on_record);
        if base.as_ref().is_some_and(|b| b.merging().is_some()) {
            return Err(SandboxError::Merging {
                branch: branch.to_string(),
            });
        }
        let made_on = base.as_ref().and_then(|b| b.commit()).cloned();
        // The conversation pinned its branch's record — origin's copy — which
        // can be ahead of the local branch the checkout is put on. When origin
        // already holds the base and the local branch is behind it, the job
        // may run: the local branch follows it once the checkout is locked and
        // set aside (`put_on`), never here, where another job may hold it.
        if let Some(made_on) = made_on.as_ref().filter(|m| tip.as_ref() != Some(*m)) {
            let behind = match &tip {
                Some(tip) => self
                    .repo
                    .is_ancestor(&Rev::Oid(tip.clone()), &Rev::Oid(made_on.clone()))
                    .map_err(|e| store(e.to_string()))?,
                None => true,
            };
            let on_record = self
                .repo
                .on_record(branch, made_on)
                .map_err(|e| store(e.to_string()))?;
            if behind && on_record {
                return Ok(made_on.clone());
            }
        }
        let named = |oid: &Option<Oid>| {
            oid.as_ref()
                .map_or_else(|| "no commit".to_string(), |o| o.to_string())
        };
        let ahead = match (&made_on, &tip) {
            (Some(made_on), Some(tip)) if made_on == tip => return Ok(tip.clone()),
            (Some(made_on), Some(tip)) => self
                .repo
                .is_ancestor(&Rev::Oid(tip.clone()), &Rev::Oid(made_on.clone()))
                .map_err(|e| store(e.to_string()))?,
            (_, None) => true,
            (None, Some(_)) => false,
        };
        let (branch, base, tip) = (branch.to_string(), named(&made_on), named(&tip));
        Err(if ahead {
            SandboxError::Ahead { branch, base, tip }
        } else {
            SandboxError::Behind { branch, base, tip }
        })
    }
}

/// Record `captured` in `files`: what the store took, and what it refused
/// with why.
fn record(
    files: &VfsStore,
    captured: Vec<(String, TimedDelta)>,
) -> (Vec<ChangedFile>, Vec<Unrecorded>) {
    let mut changed = Vec::new();
    let mut unrecorded = Vec::new();
    for (path, delta) in captured {
        match files.apply(&path, std::slice::from_ref(&delta)) {
            Ok(()) => changed.push(ChangedFile { path, delta }),
            Err(e) => unrecorded.push(Unrecorded {
                path,
                delta,
                why: e.to_string(),
            }),
        }
    }
    (changed, unrecorded)
}

/// Run `step` — blocking git and file work — off the async runtime, holding
/// the session until it has finished, even if the job awaiting it is
/// abandoned meanwhile. The session comes back with the step's result. A step
/// that fails or panics drops the session, which puts the checkout's own
/// state back before the lock is let go.
async fn off_runtime<T: Send + 'static>(
    mut session: Session,
    step: impl FnOnce(&mut Session) -> Result<T, CheckoutError> + Send + 'static,
) -> Result<(Session, T), SandboxError> {
    let (session, result) = tokio::task::spawn_blocking(move || {
        let result = step(&mut session);
        (session, result)
    })
    .await
    .map_err(|e| SandboxError::Interrupted(e.to_string()))?;
    Ok((session, result?))
}

fn same_folder(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}
