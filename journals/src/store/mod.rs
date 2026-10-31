//! The journal repository on disk (§6.2, §6.3): the tree held in memory at
//! the service's head, every write committed at once as objects, pushed to
//! origin under a lease, and — when origin moved first — replayed onto what
//! origin holds.
//!
//! The service's commits live on their own ref, [`JOURNAL_REF`], never on a
//! branch: the checkout's working tree and branches belong to the jobs that
//! run in it, and nothing here touches them. `main` on origin is the record;
//! anything merged there by hand or by a job is read at the next fetch.
//!
//! | Module | |
//! |---|---|
//! | [`read`] | a commit's files, as objects |
//! | [`diff`] | the files an operation changes |
//! | [`message`] | a commit's subject, trailers, and the operation it carries |
//! | [`sync`] | the push and fetch, run with the store free for writes |

pub mod diff;
pub mod message;
pub mod read;
pub mod sync;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use thiserror::Error;
use zend_vfs::{
    BranchName, ChangeSet, FetchSpec, GitError, GitTime, LogRange, Oid, RefName, RefOp,
    RefTransaction, RemoteName, Repo, RepoPath, Rev, Signature, ORIGIN,
};

use crate::actor::Actor;
use crate::error::Error;
use crate::line::LineId;
use crate::ops::remap::remap;
use crate::ops::{applied_to, Applied, Op};
use crate::store::diff::{apply_changes, changes, Changes};
use crate::store::message::{message, read};
use crate::store::read::files_at;
use crate::time::Stamp;
use crate::tree::load::load;
use crate::tree::Tree;

/// The ref the service's own commits are on.
pub const JOURNAL_REF: &str = "refs/journal/head";

/// The branch origin holds the record on.
pub const BRANCH: &str = "main";

/// Pushes retried against an origin that keeps moving before one lands.
const PUSH_TRIES: usize = 4;

/// The most unpushed commits a replay reads back.
const MAX_REPLAY: usize = 10_000;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error(transparent)]
    Journal(#[from] Error),
    #[error("git: {0}")]
    Git(#[from] GitError),
    #[error("{0}")]
    Repository(String),
}

/// What a sync did.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Synced {
    /// Commits origin now holds that it did not.
    pub pushed: usize,
    /// The service's commits rewritten onto a newer origin.
    pub replayed: usize,
    /// Writes origin's newer state refused when replayed: each commit's
    /// subject and why. They are gone, and the caller should say so.
    pub dropped: Vec<(String, Error)>,
    /// Lines replayed whose id another line had taken meanwhile: the id the
    /// writer was given, and the one the line has now.
    pub renamed: Vec<(LineId, LineId)>,
    /// Origin moved, and the tree was read again.
    pub fetched: bool,
    /// Origin holds everything this checkout wrote. When it kept moving
    /// through every try, this is false and the next sync goes on.
    pub settled: bool,
}

pub struct Store {
    /// Shared with a sync's push and fetch, which run without the store.
    repo: Arc<Repo>,
    branch: BranchName,
    remote: RemoteName,
    /// The service's head: what [`JOURNAL_REF`] holds.
    head: Oid,
    /// What origin's branch held when last fetched or pushed.
    origin_at: Oid,
    files: BTreeMap<String, Vec<u8>>,
    /// Shared with whoever reads it: a write makes a new tree, never changes
    /// the one handed out.
    tree: Arc<Tree>,
}

fn git_time(at: Stamp) -> GitTime {
    GitTime {
        seconds: at.secs(),
        offset_minutes: 0,
    }
}

fn journal_ref() -> RefName {
    RefName::parse(JOURNAL_REF).expect("a valid ref name")
}

impl Store {
    /// Open the checkout at `dir`: fetch origin, take up the service's own
    /// unpushed commits — replayed onto origin when it moved past them — and
    /// read the tree.
    pub fn open(dir: &Path) -> Result<Store, StoreError> {
        let repo = Repo::open(dir)?;
        let branch = BranchName::parse(BRANCH)?;
        let remote = RemoteName::parse(ORIGIN)?;
        if let Err(e) = repo.fetch(&remote, &FetchSpec::Branch(branch.clone())) {
            tracing::warn!("journals: origin could not be fetched; reading what is here: {e}");
        }
        let origin_at = repo.ref_target(&remote.tracking(&branch))?.ok_or_else(|| {
            StoreError::Repository(format!(
                "{} has no {remote}/{branch}: seed the repository first",
                dir.display()
            ))
        })?;
        let mut store = Store {
            files: files_at(&repo, &origin_at)?,
            tree: Arc::default(),
            head: origin_at.clone(),
            origin_at,
            repo: Arc::new(repo),
            branch,
            remote,
        };
        match store.repo.ref_target(&journal_ref())? {
            None => {
                let txn = RefTransaction::new().push(RefOp::Create {
                    name: journal_ref(),
                    new: store.origin_at.clone(),
                });
                store.repo.update_refs(&txn)?;
                store.tree = Arc::new(load(&store.files));
            }
            Some(head) => {
                let origin = Rev::Oid(store.origin_at.clone());
                let mine = Rev::Oid(head.clone());
                if store.repo.is_ancestor(&origin, &mine)? {
                    // Unpushed commits on top of origin: the next sync pushes them.
                    store.head = head;
                    store.files = files_at(&store.repo, &store.head)?;
                    store.tree = Arc::new(load(&store.files));
                } else {
                    // Origin moved past what this checkout last pushed:
                    // whatever of its own it still holds goes on top.
                    store.head = head;
                    let replayed = store.replay_onto(store.origin_at.clone())?;
                    tracing::info!(
                        replayed = replayed.replayed,
                        dropped = replayed.dropped.len(),
                        "journals: took up commits origin had moved past"
                    );
                    for (subject, why) in &replayed.dropped {
                        tracing::warn!("journals: dropped on replay: {subject}: {why}");
                    }
                }
            }
        }
        Ok(store)
    }

    pub fn tree(&self) -> &Tree {
        &self.tree
    }

    /// The tree, to hold beyond this borrow: no copy is made.
    pub fn shared(&self) -> Arc<Tree> {
        Arc::clone(&self.tree)
    }

    pub fn head(&self) -> &Oid {
        &self.head
    }

    /// Commits made here that origin does not hold yet.
    pub fn unpushed(&self) -> bool {
        self.head != self.origin_at
    }

    /// Apply `op` for `actor` at `now` and commit it onto the service's head.
    /// Nothing held here changes unless the commit and the ref move both do.
    pub fn write(&mut self, actor: &Actor, op: &Op, now: Stamp) -> Result<Applied, StoreError> {
        let (next, applied) = applied_to(&self.tree, actor, op, now, None)?;
        let (new, ch) = self.commit(&self.head, &self.files, &next, &applied, actor, op, now)?;
        let txn = RefTransaction::new().push(RefOp::Update {
            name: journal_ref(),
            new: new.clone(),
            old: self.head.clone(),
        });
        self.repo.update_refs(&txn)?;
        self.head = new;
        apply_changes(&mut self.files, &ch);
        self.tree = Arc::new(next);
        Ok(applied)
    }

    /// The commit of `applied` onto `base`, whose files are `before`; and
    /// the changes it made to them, for the caller to take in once the
    /// commit stands — the files are never copied whole.
    #[allow(clippy::too_many_arguments)]
    fn commit(
        &self,
        base: &Oid,
        before: &BTreeMap<String, Vec<u8>>,
        after: &Tree,
        applied: &Applied,
        actor: &Actor,
        op: &Op,
        now: Stamp,
    ) -> Result<(Oid, Changes), StoreError> {
        let ch = changes(before, after, applied);
        if ch.is_empty() {
            return Err(Error::refused("that changes nothing").into());
        }
        // The files are rendered here in their committed form, so each is
        // stored as given: no filter, and no `git` run per file.
        let mut set = ChangeSet::new();
        for (path, change) in &ch {
            let path = RepoPath::parse(path)?;
            match change {
                Some(bytes) => set.write_blob(path, bytes.clone(), None)?,
                None => set.delete(path)?,
            }
        }
        // The author's time is when the write is recorded as happening; the
        // committer's is when this service committed it, so history orders
        // by when things reached the repository whatever a write says of
        // itself.
        let writer = actor.writer().name();
        let author = Signature::new(writer, &format!("{writer}@journals.tokera"), git_time(now))?;
        let committer =
            Signature::new("journal service", "journals@tokera", git_time(Stamp::now()))?;
        let msg = message(&applied.subject, actor, op, now, applied.line.as_ref());
        let new = self
            .repo
            .commit_changes(base, &set, &msg, &author, &committer)?;
        Ok((new, ch))
    }

    /// Push what origin lacks, fetch what it has, and replay the service's
    /// commits onto it when it moved first: [`sync::sync`] on a store held
    /// by its caller alone.
    pub fn sync(&mut self) -> Result<Synced, StoreError> {
        sync::sync(self).expect("a store held by its caller is always lent")
    }

    fn count_between(&self, from: &Oid, to: &Oid) -> Result<usize, StoreError> {
        let range = LogRange {
            to: Rev::Oid(to.clone()),
            exclude: vec![Rev::Oid(from.clone())],
            paths: vec![],
        };
        Ok(self.repo.log(&range, MAX_REPLAY)?.len())
    }

    /// Rewrite the service's commits not on `onto` as fresh commits on it,
    /// each operation applied again to `onto`'s tree; one the newer tree
    /// refuses is dropped. Afterwards origin is `onto` and the head is the
    /// last rewritten commit; on a git error, nothing held here has changed.
    fn replay_onto(&mut self, onto: Oid) -> Result<Synced, StoreError> {
        let range = LogRange {
            to: Rev::Oid(self.head.clone()),
            exclude: vec![Rev::Oid(onto.clone())],
            paths: vec![],
        };
        let mut mine = self.repo.log(&range, MAX_REPLAY)?;
        mine.reverse();
        let mut files = files_at(&self.repo, &onto)?;
        let mut tree = load(&files);
        let mut base = onto.clone();
        let mut done = Synced::default();
        let mut renamed: BTreeMap<LineId, LineId> = BTreeMap::new();
        for c in mine {
            let mut rec = match read(&c.message) {
                Ok(rec) => rec,
                Err(e) => {
                    done.dropped.push((c.subject().to_string(), e));
                    continue;
                }
            };
            remap(&mut rec.op, &renamed);
            match applied_to(&tree, &rec.actor, &rec.op, rec.at, rec.line.as_ref()) {
                Ok((next, applied)) => match self
                    .commit(&base, &files, &next, &applied, &rec.actor, &rec.op, rec.at)
                {
                    Ok((new, ch)) => {
                        // Only a write of a new line mints an id; an edit or a
                        // forget names one that is already there.
                        let mints = matches!(
                            rec.op,
                            Op::Note { .. } | Op::Episode { .. } | Op::FoldPeriod { .. }
                        );
                        if let (true, Some(was), Some(is)) = (mints, &rec.line, &applied.line) {
                            if was != is {
                                renamed.insert(was.clone(), is.clone());
                                done.renamed.push((was.clone(), is.clone()));
                            }
                        }
                        base = new;
                        apply_changes(&mut files, &ch);
                        tree = next;
                        done.replayed += 1;
                    }
                    Err(StoreError::Journal(e)) => done.dropped.push((c.subject().to_string(), e)),
                    Err(e) => return Err(e),
                },
                Err(e) => done.dropped.push((c.subject().to_string(), e)),
            }
        }
        let txn = RefTransaction::new().push(RefOp::Update {
            name: journal_ref(),
            new: base.clone(),
            old: self.head.clone(),
        });
        self.repo.update_refs(&txn)?;
        self.head = base;
        self.origin_at = onto;
        self.files = files;
        self.tree = Arc::new(tree);
        Ok(done)
    }
}

#[cfg(test)]
mod tests;
