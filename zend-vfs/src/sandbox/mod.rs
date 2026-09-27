//! Running a command line program on a repository, as one conversation.
//!
//! A [`Sandbox`] is one repository's: its working folder is the repository's
//! own checkout. A conversation's changes to the repository live as deltas in
//! its [`VfsStore`], over the repository's folder; a program run on the
//! machine — a build, a test run, a formatter — needs them as files, and what
//! it changes has to come back into the store. [`Sandbox::run`] is that whole
//! round trip, one run at a time:
//!
//! 1. **Lock** the checkout. A second run on the same repository waits here.
//! 2. **Reset** the checkout — `git reset --hard` — and **switch** it to the
//!    conversation's branch when it is not on it;
//! 3. **apply** the conversation's changes onto it, writing only the files
//!    whose bytes are wrong, each dated by the conversation's latest change
//!    to it (steps 2 and 3 are [`materialize`]);
//! 4. **check** the command ([`CommandPolicy`]). Every rule that reads the
//!    command line alone — git run directly, the allow-list, paths out of the
//!    repository — is checked before step 1, so a refused command touches
//!    nothing; the one that reads the checkout — a program the conversation
//!    wrote into the repository is a plain file there — is checked here,
//!    where it now stands;
//! 5. **run** it, reading what it prints ([`process`]);
//! 6. **diff** — put the branch and `HEAD` back where step 2 left them, should
//!    the command have committed, switched or deleted the branch, and read
//!    back what it changed, commits included, as deltas against the
//!    conversation's own state ([`capture`]);
//! 7. **reset** the checkout again, so the repository's folder holds the
//!    branch and nothing of this conversation's;
//! 8. **record** the deltas in the conversation's store;
//! 9. **unlock**, and hand back what the command printed and the files it
//!    changed ([`RunOutcome`]).
//!
//! The store is the reason recording comes after the second reset. Its lower
//! layer *is* the repository's folder: a delta is checked against the file as
//! the store reads it, which is the folder's copy with the conversation's
//! earlier deltas replayed over it. Recorded while the folder still held the
//! command's output, every delta would be checked against a file it had
//! already been applied to. Reset first, the folder is the branch again and
//! the store reads exactly what the deltas were made against.
//!
//! A failure at any step after the lock still resets the checkout before the
//! lock is released, and records nothing. The checkout is the daemon's to
//! overwrite: running a sandbox on a folder where a developer keeps
//! uncommitted work destroys that work at step 2.
//!
//! | Module | Concern |
//! |---|---|
//! | [`command`] | The program, its arguments and its timeout |
//! | [`policy`] | The security check a command passes before it runs |
//! | `git_use` | Finding git run directly — as the program, or in a shell's script |
//! | [`process`] | Starting it, reading both streams, killing its process tree |
//! | [`outcome`] | What a run hands back |
//!
//! [`materialize`]: crate::checkout::materialize()
//! [`capture`]: crate::checkout::capture()

pub mod command;
mod error;
mod git_use;
pub mod outcome;
pub mod policy;
pub mod process;

use std::path::Path;
use std::sync::Arc;

use tokio::sync::{Mutex, OwnedMutexGuard};

use crate::checkout::{self, CheckoutError, Ledger};
use crate::file_delta::TimedDelta;
use crate::{BranchName, DiskWriteGrant, FileChanges, GitError, Repo, VfsStore};

pub use command::{SandboxCommand, DEFAULT_TIMEOUT};
pub use error::SandboxError;
pub use outcome::{ChangedFile, RunOutcome, Stream, Unrecorded};
pub use policy::{CommandPolicy, Refused};

/// The checkout's lock, holding what the checkout carries between runs: its
/// ledger ([`Ledger`]), `None` until a run has left one.
type Held = OwnedMutexGuard<Option<Ledger>>;

/// One repository's sandbox: its checkout, the programs it may run there, and
/// the lock that keeps runs on it one at a time.
pub struct Sandbox {
    repo: Arc<Repo>,
    policy: CommandPolicy,
    checkout: Arc<Mutex<Option<Ledger>>>,
}

impl Sandbox {
    pub fn new(repo: Repo, policy: CommandPolicy) -> Self {
        Self {
            repo: Arc::new(repo),
            policy,
            checkout: Arc::new(Mutex::new(None)),
        }
    }

    pub fn repo(&self) -> &Repo {
        &self.repo
    }

    pub fn policy(&self) -> &CommandPolicy {
        &self.policy
    }

    /// Run `command` on the repository as the conversation whose changes
    /// `files` holds, on `branch`, and record what it changed in `files`. See
    /// the module for each step.
    ///
    /// `files` must be the conversation's overlay over this repository's
    /// folder. A run abandoned part way — its future dropped — kills the
    /// command, and a checkout step already under way finishes before the
    /// lock is released; the next run puts the checkout right.
    ///
    /// **Overwrites the checkout**, which must be one the daemon owns.
    pub async fn run(
        &self,
        _grant: &DiskWriteGrant,
        branch: &BranchName,
        files: &VfsStore,
        command: &SandboxCommand,
    ) -> Result<RunOutcome, SandboxError> {
        self.check_store(files)?;
        // 4, for everything that needs no checkout: a refused command never
        // takes the lock.
        self.policy.check_command(command)?;
        // 1. Lock.
        let held = Arc::clone(&self.checkout).lock_owned().await;
        let (held, ran) = self
            .on_checkout(held, branch, files.changes(), command)
            .await?;
        // 7. Reset.
        let (held, reset) = self.put_on(held, branch, FileChanges::new()).await?;
        if let Err(reset) = reset {
            return Err(SandboxError::Reset {
                reset: Box::new(reset),
                run: ran.err().map(Box::new),
            });
        }
        let (executed, captured) = ran?;
        // 8. Record.
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
        // 9. Unlock.
        drop(held);
        Ok(RunOutcome {
            exit_code: executed.exit_code,
            timed_out: executed.timed_out,
            stdout: executed.stdout,
            stderr: executed.stderr,
            changed,
            unrecorded,
        })
    }

    /// Steps 2 to 6. Hands the lock back with how they went. Fails outright —
    /// no reset to follow — when a checkout step is lost, or when the branch
    /// does not exist: that is found before the checkout is touched, and the
    /// reset could not put the checkout on it either.
    async fn on_checkout(
        &self,
        held: Held,
        branch: &BranchName,
        changes: FileChanges,
        command: &SandboxCommand,
    ) -> Result<(Held, Result<Ran, SandboxError>), SandboxError> {
        // 2, 3. Reset, switch, apply.
        let (held, put) = self.put_on(held, branch, changes).await?;
        match put {
            Err(
                e @ SandboxError::Checkout(CheckoutError::Git(GitError::UnknownRevision { .. })),
            ) => {
                return Err(e);
            }
            Err(e) => return Ok((held, Err(e))),
            Ok(()) => {}
        }
        // 4. Check what reads the checkout.
        if let Err(refused) = self.policy.check_on_checkout(self.repo.dir(), command) {
            return Ok((held, Err(refused.into())));
        }
        // 5. Run.
        let executed = match process::execute(self.repo.dir(), command).await {
            Ok(executed) => executed,
            Err(source) => {
                let failed = SandboxError::Start {
                    program: command.program.clone(),
                    source,
                };
                return Ok((held, Err(failed)));
            }
        };
        // 6. Diff.
        let repo = Arc::clone(&self.repo);
        let branch = branch.clone();
        let (held, captured) = off_runtime(held, move |slot| {
            let mut ledger = slot.take().expect("the checkout was just put in place");
            let captured = checkout::capture(&repo, &branch, &mut ledger)?;
            *slot = Some(ledger);
            Ok(captured)
        })
        .await?;
        Ok((held, captured.map(|c| (executed, c))))
    }

    /// Put the checkout on `branch` with `changes` laid over it, carrying the
    /// ledger in the lock from one run to the next.
    async fn put_on(
        &self,
        held: Held,
        branch: &BranchName,
        changes: FileChanges,
    ) -> Result<(Held, Result<(), SandboxError>), SandboxError> {
        let repo = Arc::clone(&self.repo);
        let branch = branch.clone();
        off_runtime(held, move |slot| {
            let grant = DiskWriteGrant::issue();
            let (ledger, _) = checkout::materialize(&repo, &grant, &branch, &changes, slot.take())?;
            *slot = Some(ledger);
            Ok(())
        })
        .await
    }

    /// `files` is an overlay over this sandbox's repository.
    fn check_store(&self, files: &VfsStore) -> Result<(), SandboxError> {
        if files.is_direct() {
            return Err(SandboxError::DirectStore);
        }
        let same = files
            .root()
            .is_some_and(|root| same_folder(root, self.repo.dir()));
        if !same {
            return Err(SandboxError::WrongStore {
                store: files.root().map(Path::to_path_buf),
                repo: self.repo.dir().to_path_buf(),
            });
        }
        Ok(())
    }
}

/// What steps 5 and 6 produced: the command's run, and what it changed.
type Ran = (process::Executed, Vec<(String, TimedDelta)>);

/// Run `step` — blocking git and file work — off the async runtime, holding
/// the checkout's lock until it has finished, even if the run awaiting it is
/// abandoned meanwhile. The lock comes back with the step's result; a step
/// that panicked loses both.
async fn off_runtime<T: Send + 'static>(
    mut held: Held,
    step: impl FnOnce(&mut Option<Ledger>) -> Result<T, CheckoutError> + Send + 'static,
) -> Result<(Held, Result<T, SandboxError>), SandboxError> {
    tokio::task::spawn_blocking(move || {
        let result = step(&mut held).map_err(SandboxError::from);
        (held, result)
    })
    .await
    .map_err(|e| SandboxError::Interrupted(e.to_string()))
}

fn same_folder(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}
