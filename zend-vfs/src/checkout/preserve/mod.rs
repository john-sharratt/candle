//! Keeping a checkout's own state safe while a run uses it.
//!
//! The checkout a sandbox runs in is a repository's folder — which may well be
//! someone's working copy: a branch checked out, a staged change, edits not
//! committed, files not yet added, local files git ignores. A run puts the
//! checkout on a conversation's branch and files, which would throw all of
//! that away. So before anything is touched, [`preserve`]:
//!
//! 1. takes the checkout's lock (`lock`), so that no other process's run can
//!    meet this one;
//! 2. refuses a checkout part way through a merge, a rebase, a cherry-pick, a
//!    revert or a bisect, or on a branch with no commit yet — nothing touched;
//! 3. finishes putting back whatever an earlier run left set aside;
//! 4. writes a journal (`journal`), so that whatever happens next can be
//!    undone by whoever finds it: this run, or the next one after a crash;
//! 5. lists what is ignored (`ignored`): nothing it covers is ever removed or
//!    read back as a run's, whatever the ignore rules say later;
//! 6. captures the index and every working-tree change byte for byte as
//!    commits held by refs of the layer's own (`snapshot`) — the stash list
//!    is never touched — and every tracked file the run will rewrite, so
//!    that its line endings come back as they were;
//! 7. moves aside each ignored file the run could clobber (`aside`).
//!
//! [`Preserved::restore`] undoes it all, whether the run succeeded or not:
//! the exclude file back first, then links the run left removed without
//! being followed, `HEAD` back where it was (the branch made again at its
//! commit if the run deleted it, and put back if the run took commits off
//! it), the run's untracked files removed — never one that was ignored when
//! the state was set aside — the moved files back, and the snapshot written
//! back exactly. Every step is safe to repeat, and a failing restore is
//! tried again; one that still fails leaves the journal and the refs in
//! place and names them.
//!
//! The next [`preserve`] — or [`recover`] — finishes what a crash or a
//! failed restore left, and never over anyone's work: a checkout that has
//! moved on since, to a commit neither its own state nor the run was on, is
//! refused and left for its owner; and whatever the checkout holds is
//! captured again first, under `refs/zend/recovered/`, and kept.
//! Dropping the guard unrestored restores.

mod aside;
mod ignored;
mod journal;
mod lock;
mod raw;
mod snapshot;

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use self::journal::{Journal, Phase, Place, SavedHead};
use self::lock::CheckoutLock;
use self::snapshot::Snapshot;
use super::error::CheckoutError;
use crate::{BranchName, GitError, Head, Oid, RefName, RefOp, RefTransaction, Repo, Rev};

pub(crate) use self::ignored::Ignored;

/// What git leaves in its folder while an operation is part way through,
/// and what that operation is.
const OPERATIONS: [(&str, &str); 7] = [
    ("MERGE_HEAD", "a merge"),
    ("rebase-merge", "a rebase"),
    ("rebase-apply", "a rebase or an `am`"),
    ("CHERRY_PICK_HEAD", "a cherry-pick"),
    ("REVERT_HEAD", "a revert"),
    ("BISECT_LOG", "a bisect"),
    ("sequencer", "a series of picks"),
];

/// How many times a restore is tried, and the wait before the first retry,
/// doubled each time: a process just killed can hold its files open a
/// moment longer.
const ATTEMPTS: u32 = 5;
const FIRST_WAIT: Duration = Duration::from_millis(100);

/// A checkout's own state, set aside while a run uses the checkout. See the
/// module.
#[must_use = "dropping the guard puts the checkout back at once"]
pub struct Preserved {
    repo: Arc<Repo>,
    place: Place,
    journal: Journal,
    done: bool,
    /// The checkout's lock, for a preservation that took it; one found left
    /// on disk is put back under its finder's. Let go once the restore is
    /// over — and, declared last, only after a guard dropped unrestored has
    /// put the checkout back.
    lock: Option<CheckoutLock>,
}

impl fmt::Debug for Preserved {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Preserved")
            .field("place", &self.place.dir)
            .field("journal", &self.journal)
            .field("done", &self.done)
            .finish()
    }
}

/// Set aside `repo`'s checkout for a run that writes `writes` and puts the
/// checkout on the commit `target`. See the module. Anything a previous run
/// left set aside is put back first; when it cannot be, nothing is run.
pub fn preserve(
    repo: Arc<Repo>,
    why: &str,
    writes: &[String],
    target: Option<&Oid>,
) -> Result<Preserved, CheckoutError> {
    let lock = CheckoutLock::take(repo.git_dir())?;
    refuse_mid_operation(&repo)?;
    recover_locked(&repo)?;
    let (branch, commit) = match repo.head()? {
        Head::Branch { branch, oid } => (Some(branch.as_str().to_string()), oid),
        Head::Detached(oid) => (None, oid),
        Head::Unborn(branch) => {
            return Err(GitError::invalid(format!(
                "the checkout is on {branch}, which has no commit yet, so its files cannot be \
                 set aside safely; nothing was touched"
            ))
            .into())
        }
    };
    let landing = aside::landing(&repo, writes, target)?;
    let place = Place::new(repo.git_dir());
    let mut journal = Journal {
        why: why.to_string(),
        phase: Phase::Capturing,
        head: SavedHead {
            branch,
            commit: commit.as_str().to_string(),
        },
        target: target.map(|t| t.as_str().to_string()),
        index: None,
        files: None,
        deleted: Vec::new(),
        flags: Vec::new(),
        intent_to_add: Vec::new(),
        perms: Default::default(),
        link_dirs: Vec::new(),
        exclude: None,
        vacant: Vec::new(),
        moved: Vec::new(),
    };
    journal.set_exclude(read_exclude(&repo)?.as_deref());
    place.save(&journal)?;
    // From here the guard exists: a failure puts back whatever was done.
    let mut preserved = Preserved {
        repo,
        place,
        journal,
        done: false,
        lock: Some(lock),
    };
    preserved
        .place
        .save_ignored(&Ignored::listed(&preserved.repo)?)?;

    let snapshot = snapshot::capture(&preserved.repo, &commit, why, &landing.tracked)?;
    preserved.hold(&snapshot)?;
    preserved.journal.index = Some(snapshot.index.as_str().to_string());
    preserved.journal.files = snapshot.files.as_ref().map(|f| f.as_str().to_string());
    preserved.journal.deleted = snapshot.deleted;
    preserved.journal.flags = snapshot.flags;
    preserved.journal.intent_to_add = snapshot.intent_to_add;
    preserved.journal.perms = snapshot.perms;
    preserved.journal.link_dirs = snapshot.link_dirs;
    preserved.place.save(&preserved.journal)?;

    aside::set_aside(
        &preserved.repo,
        &preserved.place,
        &mut preserved.journal,
        &landing.untracked,
    )?;
    preserved.journal.phase = Phase::Preserved;
    preserved.place.save(&preserved.journal)?;
    Ok(preserved)
}

/// Put back everything an earlier run left set aside in `repo`'s checkout —
/// a run that crashed, or whose restore failed. An error means something is
/// still set aside, and names where.
pub fn recover(repo: &Arc<Repo>) -> Result<(), CheckoutError> {
    let _lock = CheckoutLock::take(repo.git_dir())?;
    refuse_mid_operation(repo)?;
    recover_locked(repo)
}

/// A checkout part way through an operation of git's is its owner's to
/// finish: refused, nothing touched.
fn refuse_mid_operation(repo: &Repo) -> Result<(), CheckoutError> {
    match OPERATIONS
        .iter()
        .find(|(marker, _)| repo.git_dir().join(marker).exists())
    {
        Some((_, what)) => Err(GitError::invalid(format!(
            "the checkout is part way through {what}; nothing was touched — finish or abort \
             it first"
        ))
        .into()),
        None => Ok(()),
    }
}

/// [`recover`], with the checkout's lock held: every preservation on disk
/// is one a run that is gone left behind.
fn recover_locked(repo: &Arc<Repo>) -> Result<(), CheckoutError> {
    for place in Place::left_in(repo.git_dir())? {
        let journal = place.load()?;
        let mut left = Preserved {
            repo: Arc::clone(repo),
            place,
            journal,
            done: false,
            lock: None,
        };
        left.recover_one()?;
    }
    Ok(())
}

impl Preserved {
    /// Put the checkout back as it was when preserved. See the module. Tried
    /// until it succeeds or the attempts run out; either way the guard does
    /// nothing more when dropped.
    pub fn restore(&mut self) -> Result<(), CheckoutError> {
        if self.done {
            return Ok(());
        }
        self.done = true;
        let mut wait = FIRST_WAIT;
        let mut last = String::new();
        for attempt in 1..=ATTEMPTS {
            match self.put_back() {
                Ok(()) => {
                    self.lock = None;
                    return Ok(());
                }
                Err(e) => last = e.to_string(),
            }
            if attempt < ATTEMPTS {
                std::thread::sleep(wait);
                wait *= 2;
            }
        }
        self.lock = None;
        Err(self.not_put_back(last))
    }

    fn not_put_back(&self, detail: String) -> CheckoutError {
        CheckoutError::NotPutBack {
            journal: self.place.dir.to_string_lossy().into_owned(),
            detail,
        }
    }

    /// Finish what a run that is gone left: see the module.
    fn recover_one(&mut self) -> Result<(), CheckoutError> {
        match self.journal.phase {
            Phase::Restored => {}
            Phase::Capturing => {
                // Nothing but moving files aside had begun — and someone may
                // have put a file where one of them goes since.
                let taken = aside::taken_homes(&self.repo, &self.place, &self.journal)?;
                if !taken.is_empty() {
                    self.done = true;
                    return Err(self.not_put_back(format!(
                        "files kept aside belong where something else stands now: {}",
                        taken.join(", ")
                    )));
                }
            }
            Phase::Preserved | Phase::Restoring => {
                if self.journal.phase == Phase::Preserved {
                    self.refuse_moved_on()?;
                }
                self.keep_what_stands()?;
            }
        }
        self.restore()
    }

    /// A checkout a run left mid-way holds the owner's commit or the run's.
    /// On any other, someone has worked in it since: refused, and left for
    /// them, with what is set aside named.
    fn refuse_moved_on(&mut self) -> Result<(), CheckoutError> {
        let now = match self.repo.head()? {
            Head::Branch { oid, .. } | Head::Detached(oid) => oid,
            Head::Unborn(_) => {
                self.done = true;
                return Err(self.not_put_back(
                    "the checkout is on a branch with no commit now, so someone has worked in \
                     it since the run; nothing was put back"
                        .to_string(),
                ));
            }
        };
        let known = [
            Some(&self.journal.head.commit),
            self.journal.target.as_ref(),
        ];
        if known.iter().flatten().any(|c| c.as_str() == now.as_str()) {
            return Ok(());
        }
        self.done = true;
        Err(self.not_put_back(format!(
            "the checkout is at {now} now, which neither it nor the run was on, so someone has \
             worked in it since the run; nothing was put back. What was set aside is kept at \
             refs/zend/preserved/{} and in this folder; remove the folder once it is no longer \
             needed",
            self.place.id()
        )))
    }

    /// Whatever the checkout holds now — including anything standing where
    /// files kept aside go back — captured and kept under
    /// `refs/zend/recovered/`, before a restore replaces it.
    fn keep_what_stands(&self) -> Result<(), CheckoutError> {
        let head = match self.repo.head()? {
            Head::Branch { oid, .. } | Head::Detached(oid) => oid,
            Head::Unborn(_) => return Ok(()),
        };
        let id = self.place.id();
        let index_ref = RefName::parse(&format!("refs/zend/recovered/{id}/index"))?;
        // A recovery tried before kept what stood then — before any of it
        // was put back — and that is the one worth keeping.
        if self.repo.ref_target(&index_ref)?.is_some() {
            return Ok(());
        }
        let also: Vec<String> = self
            .journal
            .vacant
            .iter()
            .chain(&self.journal.moved)
            .cloned()
            .collect();
        let taken = snapshot::capture(
            &self.repo,
            &head,
            &format!("recovering what a run left ({id})"),
            &also,
        )?;
        let mut txn = RefTransaction::new().push(RefOp::Create {
            name: index_ref,
            new: taken.index,
        });
        if let Some(files) = taken.files {
            txn = txn.push(RefOp::Create {
                name: RefName::parse(&format!("refs/zend/recovered/{id}/files"))?,
                new: files,
            });
        }
        self.repo.update_refs(&txn)?;
        tracing::warn!(
            "a run left this checkout set aside; putting it back — what it held first is kept \
             at refs/zend/recovered/{id}"
        );
        Ok(())
    }

    fn put_back(&mut self) -> Result<(), CheckoutError> {
        match self.journal.phase {
            Phase::Restored => {}
            Phase::Capturing => {
                // Nothing but moving files aside had begun.
                aside::put_back(&self.repo, &self.place, &mut self.journal, false)?;
            }
            Phase::Preserved | Phase::Restoring => {
                if self.journal.phase == Phase::Preserved {
                    self.journal.phase = Phase::Restoring;
                    self.place.save(&self.journal)?;
                }
                // The ignore rules first — the exclude file here, `.gitignore`
                // with the checkout of `HEAD` — and what was ignored when the
                // state was set aside is never the run's, whatever they say.
                // Links go before the checkout, which would write through
                // them, and the run's untracked files only after it.
                write_exclude(&self.repo, self.journal.exclude_bytes()?.as_deref())?;
                let kept = Ignored::kept_in(&self.repo)?;
                self.unlink_the_run(&kept)?;
                self.check_out_head()?;
                self.clear_the_run(&kept)?;
                aside::put_back(&self.repo, &self.place, &mut self.journal, true)?;
                if let Some(snapshot) = self.snapshot()? {
                    snapshot::restore(&self.repo, &snapshot)?;
                }
            }
        }
        self.journal.phase = Phase::Restored;
        self.place.save(&self.journal)?;
        self.release()
    }

    /// Every link the run left, gone — never followed: git's own checkout
    /// writes through a link standing where a folder was, outside the
    /// repository if that is where it points.
    fn unlink_the_run(&self, kept: &Ignored) -> Result<(), CheckoutError> {
        let root = self.repo.dir();
        for rel in raw::untracked(&self.repo)? {
            if !kept.covers(rel.strip_suffix(b"/").unwrap_or(&rel)) {
                raw::remove_link(root, &rel)?;
            }
        }
        for rel in raw::changed_tracked(&self.repo)? {
            raw::clear_the_way(root, &rel)?;
        }
        Ok(())
    }

    /// Every untracked file the run left, and every link, gone — never
    /// followed — except what was ignored when the state was set aside.
    fn clear_the_run(&self, kept: &Ignored) -> Result<(), CheckoutError> {
        let root = self.repo.dir();
        for rel in raw::untracked(&self.repo)? {
            raw::remove_untracked(root, &rel, kept)?;
        }
        for rel in raw::changed_tracked(&self.repo)? {
            raw::clear_the_way(root, &rel)?;
        }
        Ok(())
    }

    /// `HEAD` back where it was: on its branch — made again at the commit it
    /// was at should the run have deleted it, and put back there should the
    /// run have taken commits off it — or detached at its commit. A branch
    /// that only moved on from that commit keeps what it gained: a commit
    /// published meanwhile moves it so.
    fn check_out_head(&self) -> Result<(), CheckoutError> {
        let repo = &self.repo;
        let commit = Oid::parse(&self.journal.head.commit)?;
        let target: Vec<String> = match &self.journal.head.branch {
            Some(name) => {
                let branch = BranchName::parse(name)?;
                match repo.ref_target(&branch.to_ref())? {
                    None => repo.force_branch_tip(&branch, None, &commit)?,
                    Some(now)
                        if now != commit
                            && !repo.is_ancestor(
                                &Rev::Oid(commit.clone()),
                                &Rev::Oid(now.clone()),
                            )? =>
                    {
                        repo.force_branch_tip(&branch, Some(&now), &commit)?
                    }
                    Some(_) => {}
                }
                vec![name.clone()]
            }
            None => vec!["--detach".to_string(), commit.as_str().to_string()],
        };
        let _write = repo.write_lock();
        repo.git("checkout")
            .args(["--quiet", "--force", "--no-recurse-submodules"])
            .args(target)
            .arg("--")
            .run_ok()?;
        Ok(())
    }

    fn snapshot(&self) -> Result<Option<Snapshot>, CheckoutError> {
        let Some(index) = &self.journal.index else {
            return Ok(None);
        };
        Ok(Some(Snapshot {
            index: Oid::parse(index)?,
            files: self.journal.files.as_deref().map(Oid::parse).transpose()?,
            deleted: self.journal.deleted.clone(),
            flags: self.journal.flags.clone(),
            intent_to_add: self.journal.intent_to_add.clone(),
            perms: self.journal.perms.clone(),
            link_dirs: self.journal.link_dirs.clone(),
        }))
    }

    /// The refs that keep this preservation's commits from being pruned.
    fn refs(&self) -> Result<[RefName; 2], CheckoutError> {
        let id = self.place.id();
        Ok([
            RefName::parse(&format!("refs/zend/preserved/{id}/index"))?,
            RefName::parse(&format!("refs/zend/preserved/{id}/files"))?,
        ])
    }

    fn hold(&self, snapshot: &Snapshot) -> Result<(), CheckoutError> {
        let [index, files] = self.refs()?;
        let mut txn = RefTransaction::new().push(RefOp::Create {
            name: index,
            new: snapshot.index.clone(),
        });
        if let Some(oid) = &snapshot.files {
            txn = txn.push(RefOp::Create {
                name: files,
                new: oid.clone(),
            });
        }
        self.repo.update_refs(&txn)?;
        Ok(())
    }

    /// The refs and the folder, gone: nothing is left to put back. Only
    /// once the journal says the checkout is back, so that a crash in
    /// between leaves nothing to undo.
    fn release(&self) -> Result<(), CheckoutError> {
        let mut txn = RefTransaction::new();
        for name in self.refs()? {
            if let Some(old) = self.repo.ref_target(&name)? {
                txn = txn.push(RefOp::Delete { name, old });
            }
        }
        self.repo.update_refs(&txn)?;
        self.place.remove()
    }
}

#[cfg(test)]
impl Preserved {
    /// What a crash leaves: the journal and everything set aside on disk,
    /// the lock let go as the operating system lets go of a dead process's.
    pub(super) fn crash(mut self) {
        self.done = true;
    }
}

impl Drop for Preserved {
    /// A run that did not restore the checkout itself — it failed, panicked
    /// or was abandoned — restores it here. A failure can only be reported:
    /// what is still set aside stays, journalled, for the next run to finish.
    fn drop(&mut self) {
        if let Err(e) = self.restore() {
            tracing::error!("the checkout could not be put back as it was: {e}");
        }
    }
}

/// `.git/info/exclude` as it stands; `None` when there is none.
fn read_exclude(repo: &Repo) -> Result<Option<Vec<u8>>, CheckoutError> {
    let path = repo.git_dir().join("info").join("exclude");
    match std::fs::read(&path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(CheckoutError::io("info/exclude", e)),
    }
}

/// `.git/info/exclude` put back as it was: the ignore rules the run's
/// untracked files are removed under are the checkout's own, whatever the
/// run did to them.
fn write_exclude(repo: &Repo, bytes: Option<&[u8]>) -> Result<(), CheckoutError> {
    let path = repo.git_dir().join("info").join("exclude");
    let fail = |e| CheckoutError::io("info/exclude", e);
    match bytes {
        Some(bytes) => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(fail)?;
            }
            std::fs::write(&path, bytes).map_err(fail)
        }
        None => match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(fail(e)),
        },
    }
}

#[cfg(test)]
mod tests;
